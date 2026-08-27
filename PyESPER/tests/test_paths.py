"""Tests for :mod:`PyESPER.paths` -- data-directory resolution without a caller path."""

import numpy as np
import pytest

from PyESPER import paths

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_explicit_path_wins(monkeypatch):
    monkeypatch.setenv("PYESPER_DATA_DIR", "/somewhere/else")
    assert paths.data_root("/explicit") == "/explicit"


def test_env_var_used_when_valid(monkeypatch, tmp_path):
    (tmp_path / "Mat_fullgrid").mkdir()
    monkeypatch.setenv("PYESPER_DATA_DIR", str(tmp_path))
    assert paths.data_root() == str(tmp_path)
    assert paths.data_root("") == str(tmp_path)  # empty string == not given


def test_env_var_without_data_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PYESPER_DATA_DIR", str(tmp_path))  # no Mat_fullgrid inside
    with pytest.raises(FileNotFoundError, match="PYESPER_DATA_DIR"):
        paths.data_root()


@pytest.mark.skipif(
    not (REPO_ROOT / "Mat_fullgrid").is_dir(), reason="repo data not present"
)
def test_autodetect_finds_the_repo(monkeypatch):
    """The package's parent directory holds the data in a checkout or an editable
    install -- the configuration that lets callers omit the path entirely."""
    monkeypatch.delenv("PYESPER_DATA_DIR", raising=False)
    assert paths.data_root() == str(REPO_ROOT)


@pytest.mark.skipif(
    not (REPO_ROOT / "Mat_fullgrid").is_dir(), reason="repo data not present"
)
def test_estimation_works_with_no_path(monkeypatch):
    """End to end: both engines run with path omitted, matching explicit-path runs."""
    monkeypatch.delenv("PYESPER_DATA_DIR", raising=False)
    from PyESPER.lir import lir
    from PyESPER.nn import nn

    n = 60
    rng = np.random.default_rng(0)
    coords = {
        "longitude": rng.uniform(0, 360, n),
        "latitude": rng.uniform(-70, 70, n),
        "depth": rng.uniform(0, 4000, n),
    }
    preds = {
        "salinity": rng.uniform(32, 36, n),
        "temperature": rng.uniform(-1, 25, n),
    }
    kwargs = dict(
        EstDates=np.full(n, 2002.0), Equations=[8], verbose=False,
        compute_uncertainties=False,
    )
    est_l_nopath, _c, _u = lir(["TA"], "", coords, preds, **kwargs)
    est_l_path, _c, _u = lir(["TA"], str(REPO_ROOT), coords, preds, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(est_l_nopath["TA8"], dtype=float),
        np.asarray(est_l_path["TA8"], dtype=float),
    )

    est_n_nopath, _u = nn(["TA"], "", coords, preds, **kwargs)
    est_n_path, _u = nn(["TA"], str(REPO_ROOT), coords, preds, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(est_n_nopath["TA8"], dtype=float),
        np.asarray(est_n_path["TA8"], dtype=float),
    )
