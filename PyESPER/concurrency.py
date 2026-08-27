"""Concurrency policy for PyESPER's numba kernels.

Why this module exists
----------------------
PyESPER's hot paths are numba ``@njit(parallel=True)`` kernels. Each one claims the
*entire* numba thread pool (``numba.config.NUMBA_NUM_THREADS``, which defaults to every
core the process can see) for the duration of the call. That is the right thing to do
when a single call owns the machine, and the wrong thing to do when several callers
enter concurrently.

The gridded entry points in :mod:`PyESPER.xr_methods` are driven by dask, and dask
decides its own scheduler at ``.compute()`` time -- in the caller's code, not ours,
often in a different package entirely. Under dask's default threaded scheduler, several
chunks enter ``_estimate_block`` at once, each spawning a full-width numba parallel
region. That was observed to reliably **deadlock** the whole process (every thread
parked in a futex wait, 0% CPU, stuck at a fixed completed-chunk count) rather than
merely thrash, and to multiply peak memory by the worker count.

The previous mitigation was documentation: a note telling callers to force
``scheduler="synchronous"``. That is not enforceable -- PyESPER returns *lazy* arrays
and never sees the compute call -- and in practice the caller does not do it.

What this module does instead
-----------------------------
It makes concurrent entry **safe by construction**, from inside PyESPER, with no
cooperation required from any caller:

* :func:`kernel_lock` serialises entry into the kernels with a module-level semaphore.
  Whichever scheduler shows up, exactly one chunk is ever inside a numba parallel
  region, and that chunk gets every core. The other callers block harmlessly on the
  semaphore (which releases the GIL), so throughput is unchanged -- the work was never
  going to overlap usefully anyway, since each chunk already saturates the machine.
* It sets ``numba.set_num_threads`` *inside* the lock, from the calling thread.
  ``set_num_threads`` is **thread-local**; setting it once from the main thread has no
  effect on dask's worker threads, which is a very easy mistake to make here.
* Holding the lock across the whole block also serialises numba's first-call JIT
  compilation, which removes the concurrent-compile hazard for free.

Peak memory becomes a deterministic one-chunk bound rather than
``n_workers x chunk``.

Scope and limits
----------------
The semaphore is **per process**. It fully covers the common case (one process, dask's
threaded scheduler, several worker threads) and the ``distributed``-with-threads case.
It does *not* coordinate across separate processes: N worker processes on one node will
each claim the thread count resolved by :func:`num_threads`, and their memory adds up.
Under ``distributed`` that count is divided by the worker's ``nthreads``, but not by the
number of worker processes on the node -- set ``PYESPER_NUM_THREADS`` explicitly if you
run several PyESPER processes side by side.

Escape hatches
--------------
``PYESPER_NUM_THREADS``
    Explicit numba thread count for the kernels. Overrides all detection below.
``PYESPER_SERIALIZE_KERNELS=0``
    Disable the semaphore. Only correct if every kernel PyESPER reaches is compiled
    ``parallel=False``, or if the caller has genuinely guaranteed one chunk at a time.
    The deadlock described above is what this protects against; turn it off knowingly.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading

logger = logging.getLogger(__name__)

# One permit: at most one thread inside a PyESPER numba kernel at a time.
# BoundedSemaphore (not Semaphore) so an unbalanced release raises instead of silently
# raising the limit.
_KERNEL_SEMAPHORE = threading.BoundedSemaphore(1)

# Guards the memoised thread-count decision and the one-time log line.
_POLICY_LOCK = threading.Lock()
_RESOLVED_THREADS: int | None = None
_LOGGED = False

# Re-entrancy guard. The semaphore itself is not re-entrant, so a nested acquire on the
# same thread would deadlock against itself. Nothing nests today, but the lock is meant
# to be pushed further down the call stack over time (into the kernels themselves), and
# that refactor should not be able to hang the process. A nested ``kernel_lock()`` is a
# no-op: the outermost one already holds the permit and already set the thread count.
_HELD = threading.local()


def _serialize_enabled() -> bool:
    return os.environ.get("PYESPER_SERIALIZE_KERNELS", "1").strip() not in ("0", "false", "False")


def _available_cpus() -> int:
    """CPUs this process may actually run on.

    ``sched_getaffinity`` respects cgroup/taskset pinning (Slurm, containers);
    ``os.cpu_count()`` does not and will happily report every core on the node.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # not Linux, or affinity unavailable
        return os.cpu_count() or 1


def _distributed_worker_threads() -> int | None:
    """``nthreads`` of the ``distributed`` worker running this thread, else ``None``.

    Under ``distributed`` the semaphore is per-process, so a worker configured with
    several threads can still have only one of them inside the kernel -- but the other
    processes on the node are genuinely concurrent, so the per-process share of the
    machine is what we want to claim.
    """
    try:
        from distributed import get_worker
    except ImportError:
        return None
    try:
        worker = get_worker()
    except ValueError:
        return None  # not inside a worker
    return getattr(worker, "nthreads", None)


def num_threads() -> int:
    """Resolve (and memoise) how many numba threads the kernels should use."""
    global _RESOLVED_THREADS
    resolved = _RESOLVED_THREADS
    if resolved is not None:
        return resolved

    with _POLICY_LOCK:
        if _RESOLVED_THREADS is not None:
            return _RESOLVED_THREADS

        import numba

        env = os.environ.get("PYESPER_NUM_THREADS", "").strip()
        if env:
            try:
                requested = int(env)
            except ValueError:
                raise ValueError(
                    f"PYESPER_NUM_THREADS must be a positive integer, got {env!r}."
                ) from None
            if requested < 1:
                raise ValueError(
                    f"PYESPER_NUM_THREADS must be >= 1, got {requested}."
                )
            # numba refuses set_num_threads above the pool size fixed at import.
            resolved = min(requested, numba.config.NUMBA_NUM_THREADS)
        else:
            cpus = _available_cpus()
            per_worker = _distributed_worker_threads()
            if per_worker:
                cpus = max(1, cpus // per_worker)
            resolved = max(1, min(cpus, numba.config.NUMBA_NUM_THREADS))

        _RESOLVED_THREADS = resolved
        return resolved


def _log_policy_once(threads: int) -> None:
    global _LOGGED
    if _LOGGED:
        return
    with _POLICY_LOCK:
        if _LOGGED:
            return
        _LOGGED = True
    import numba

    try:
        layer = numba.threading_layer()
    except Exception:  # noqa: BLE001 -- layer is only resolved after a parallel call
        layer = "unresolved"
    logger.info(
        "PyESPER kernels: %d numba threads, threading layer %r, serialised=%s",
        threads,
        layer,
        _serialize_enabled(),
    )


@contextlib.contextmanager
def kernel_lock():
    """Serialise entry into PyESPER's numba kernels and claim the thread pool.

    Wrap the *whole* unit of work, not just the kernel call -- that is what makes peak
    memory a deterministic one-chunk bound rather than one-chunk-times-workers.

    Yields the numba thread count in effect inside the block.
    """
    if getattr(_HELD, "active", False):
        # Already inside a kernel_lock on this thread -- nothing to acquire, and the
        # thread count is already set. Yield it unchanged.
        import numba

        yield numba.get_num_threads()
        return

    acquired = _serialize_enabled()
    if acquired:
        _KERNEL_SEMAPHORE.acquire()
    try:
        import numba

        threads = num_threads()
        # Thread-local in numba: this call affects *this* thread only, which is exactly
        # why it has to happen here rather than once at import from the main thread.
        previous = numba.get_num_threads()
        changed = threads != previous
        if changed:
            numba.set_num_threads(threads)
        _log_policy_once(threads)
        _HELD.active = True
        try:
            yield threads
        finally:
            _HELD.active = False
            if changed:
                numba.set_num_threads(previous)
    finally:
        if acquired:
            _KERNEL_SEMAPHORE.release()


def warmup() -> None:
    """Compile the numba kernels now, on tiny inputs.

    Call this once at process start if you would rather pay JIT compilation up front
    than inside the first (already long) chunk. Safe to call repeatedly and from
    multiple threads -- compilation happens inside :func:`kernel_lock`, so exactly one
    thread compiles.
    """
    import numpy as np

    with kernel_lock():
        import PyESPER.eos80_jit as sw
        from PyESPER.kernels import warmup_nn

        depth = np.array([0.0, 10.0], dtype=np.float64)
        lat = np.array([0.0, 45.0], dtype=np.float64)
        sal = np.array([35.0, 34.0], dtype=np.float64)
        temp = np.array([10.0, 12.0], dtype=np.float64)
        pressure = sw.pres(depth, lat)
        sw.ptmp(sal, temp, pressure, 0.0)
        sw.satO2(sal, temp)
        sw.dens(sal, temp, pressure)
        warmup_nn()
