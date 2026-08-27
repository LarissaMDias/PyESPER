"""Tests for the kernel serialisation policy in :mod:`PyESPER.concurrency`.

Background
----------
Every hot path in PyESPER goes through numba ``@njit(parallel=True)`` kernels
(``run_nets._tansig``, and the ``eos80_jit`` seawater routines that both the LIR and NN
paths call). Each claims the whole numba thread pool per call. Several dask worker
threads entering them at once was observed to deadlock the process outright -- every
thread in a futex wait at 0% CPU -- as well as multiplying peak memory by the worker
count.

``PyESPER.concurrency.kernel_lock`` fixes that from inside PyESPER, rather than by
asking callers to pick a particular dask scheduler (which is not enforceable: these
entry points return lazy arrays and never see the ``.compute()`` call).

The load-bearing property is simply **mutual exclusion**, which these tests assert
directly. Reproducing the deadlock itself is not a reliable test -- it is a race -- so
the last test here checks the observable consequence instead: a multi-chunk estimate
under dask's *threaded* scheduler completes, and agrees with the same estimate computed
one chunk at a time.
"""

import os
import threading
import time

import numpy as np
import pytest

from PyESPER import concurrency


@pytest.fixture(autouse=True)
def _reset_policy(monkeypatch):
    """Clear the memoised thread-count decision around each test."""
    monkeypatch.setattr(concurrency, "_RESOLVED_THREADS", None, raising=False)
    yield
    concurrency._RESOLVED_THREADS = None


def test_kernel_lock_is_mutually_exclusive():
    """No two threads may be inside ``kernel_lock`` simultaneously.

    This is the whole point of the module: it is what turns "8 dask workers each
    spawning a full-width numba parallel region" into "one at a time".
    """
    n_threads = 16
    inside = 0
    peak = 0
    bookkeeping = threading.Lock()
    start = threading.Barrier(n_threads)

    def worker():
        nonlocal inside, peak
        start.wait()
        with concurrency.kernel_lock():
            with bookkeeping:
                inside += 1
                peak = max(peak, inside)
            # Hold long enough that an unserialised implementation would reliably
            # overlap; short enough to keep the test fast.
            time.sleep(0.01)
            with bookkeeping:
                inside -= 1

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not any(t.is_alive() for t in threads), "kernel_lock deadlocked"
    assert inside == 0
    assert peak == 1, f"{peak} threads were inside the kernel lock at once"


def test_kernel_lock_releases_on_exception():
    """An exception inside the block must not leak the permit."""
    with pytest.raises(RuntimeError):
        with concurrency.kernel_lock():
            raise RuntimeError("boom")

    # If the permit leaked, this would block forever.
    done = threading.Event()

    def worker():
        with concurrency.kernel_lock():
            done.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)
    assert done.is_set(), "kernel_lock permit was not released after an exception"


def test_kernel_lock_can_be_disabled(monkeypatch):
    """``PYESPER_SERIALIZE_KERNELS=0`` is a real escape hatch, and is off by default.

    Without it there would be no way to demonstrate that the serialisation is what
    provides mutual exclusion, nor to opt out once every kernel is compiled
    ``parallel=False``.
    """
    monkeypatch.setenv("PYESPER_SERIALIZE_KERNELS", "0")
    assert concurrency._serialize_enabled() is False

    n_threads = 8
    inside = 0
    peak = 0
    bookkeeping = threading.Lock()
    start = threading.Barrier(n_threads)

    def worker():
        nonlocal inside, peak
        start.wait()
        with concurrency.kernel_lock():
            with bookkeeping:
                inside += 1
                peak = max(peak, inside)
            time.sleep(0.02)
            with bookkeeping:
                inside -= 1

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert peak > 1, (
        "with serialisation disabled the threads should overlap; if they did not, "
        "this test is not actually exercising the lock"
    )

    monkeypatch.delenv("PYESPER_SERIALIZE_KERNELS")
    assert concurrency._serialize_enabled() is True


def test_num_threads_honours_env(monkeypatch):
    import numba

    monkeypatch.setenv("PYESPER_NUM_THREADS", "3")
    concurrency._RESOLVED_THREADS = None
    assert concurrency.num_threads() == min(3, numba.config.NUMBA_NUM_THREADS)


@pytest.mark.parametrize("bad", ["0", "-1", "not-a-number"])
def test_num_threads_rejects_bad_env(monkeypatch, bad):
    monkeypatch.setenv("PYESPER_NUM_THREADS", bad)
    concurrency._RESOLVED_THREADS = None
    with pytest.raises(ValueError):
        concurrency.num_threads()


def test_num_threads_is_positive_and_within_pool():
    import numba

    n = concurrency.num_threads()
    assert 1 <= n <= numba.config.NUMBA_NUM_THREADS


def test_kernel_lock_restores_the_callers_thread_count(monkeypatch):
    """``numba.set_num_threads`` is thread-local; the lock must leave it as it found it.

    A caller that deliberately pinned numba to a small count (e.g. for a reproducible
    benchmark) must still see that count after PyESPER returns.
    """
    import numba

    monkeypatch.setenv("PYESPER_NUM_THREADS", "2")
    concurrency._RESOLVED_THREADS = None

    original = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        with concurrency.kernel_lock() as threads:
            assert threads == min(2, numba.config.NUMBA_NUM_THREADS)
            assert numba.get_num_threads() == threads
        assert numba.get_num_threads() == 1
    finally:
        numba.set_num_threads(original)


@pytest.mark.slow
def test_nn_xr_completes_under_dasks_threaded_scheduler():
    """The end-to-end consequence: a multi-chunk NN estimate completes and is correct.

    Before ``kernel_lock``, several chunks entering ``run_nets`` concurrently under
    dask's default threaded scheduler could park the whole process. This runs a
    genuinely multi-chunk problem through that exact scheduler with a wall-clock bound,
    and checks the answer against the same computation done one chunk at a time.

    ``max_points_per_chunk`` is forced low: at the automatic cap the rechunk in
    ``_estimate_xr`` would leave a test-sized grid as a single block, and a single block
    cannot exercise concurrency at all.
    """
    dask = pytest.importorskip("dask")
    xr = pytest.importorskip("xarray")

    from PyESPER.xr_methods import nn_xr

    rng = np.random.default_rng(0)
    n_z, n_y = 8, 128
    dims = ("z", "y")

    def da(values):
        return xr.DataArray(values, dims=dims).chunk({"z": 1, "y": 32})

    salinity = da(rng.uniform(32.0, 36.0, (n_z, n_y)))
    temperature = da(rng.uniform(-1.0, 25.0, (n_z, n_y)))
    longitude = da(rng.uniform(0.0, 360.0, (n_z, n_y)))
    latitude = da(rng.uniform(-70.0, 70.0, (n_z, n_y)))
    depth = da(rng.uniform(0.0, 4000.0, (n_z, n_y)))

    kwargs = dict(
        variables=["TA", "oxygen"], path="", equation=8, est_dates=2002.0,
        max_points_per_chunk=64,
    )
    lazy = nn_xr(salinity, temperature, longitude, latitude, depth, **kwargs)
    assert all(v.chunks is not None for v in lazy.values()), "results should stay lazy"
    n_blocks = lazy["TA"].data.npartitions
    assert n_blocks >= 8, (
        f"only {n_blocks} block(s); this test needs several concurrent chunks to mean "
        "anything"
    )

    started = time.monotonic()
    with dask.config.set(scheduler="threads", num_workers=8):
        threaded = {k: v.compute() for k, v in lazy.items()}
    elapsed = time.monotonic() - started
    assert elapsed < 600, f"threaded compute took {elapsed:.0f}s (expected seconds)"

    lazy_ref = nn_xr(salinity, temperature, longitude, latitude, depth, **kwargs)
    with dask.config.set(scheduler="synchronous"):
        serial = {k: v.compute() for k, v in lazy_ref.items()}

    for var in threaded:
        np.testing.assert_allclose(
            threaded[var].values,
            serial[var].values,
            rtol=1e-12,
            atol=0.0,
            err_msg=f"{var}: threaded and serial schedulers disagree",
        )


def test_kernel_lock_is_reentrant_on_one_thread():
    """A nested acquire on the same thread must not deadlock against itself.

    Nothing nests today, but the lock is intended to migrate down the call stack as the
    kernels are rewritten, and that refactor must not be able to hang the process.
    """
    import numba

    done = []

    def nested():
        with concurrency.kernel_lock() as outer:
            with concurrency.kernel_lock() as inner:
                assert inner == outer == numba.get_num_threads()
                done.append(True)

    t = threading.Thread(target=nested)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "nested kernel_lock deadlocked"
    assert done == [True]

    # And the permit is still available afterwards.
    released = threading.Event()

    def after():
        with concurrency.kernel_lock():
            released.set()

    t2 = threading.Thread(target=after)
    t2.start()
    t2.join(timeout=10)
    assert released.is_set(), "permit not released after nested use"
