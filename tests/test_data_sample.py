from types import SimpleNamespace

import numpy as np
import pytest

from fmask.lightgbmlib import LightGBM
from fmask.satellite import Data

BANDS = ["blue", "green", "red", "nir"]


def _data():
    cube = np.random.default_rng(0).random((4, 30, 40)).astype(np.float32)
    return Data(cube, BANDS)


def _coords():
    rng = np.random.default_rng(1)
    rows, cols = rng.integers(0, 30, 200), rng.integers(0, 40, 200)
    rows[:20], cols[:20] = rows[20:40], cols[20:40]
    return rows, cols


def test_sample_matches_get():
    data = _data()
    bands = ["nir", "blue", "red"]
    rows, cols = _coords()
    expected = data.get(bands)[:, rows, cols]
    actual = data.sample(bands, rows, cols)
    assert actual.dtype == expected.dtype
    assert actual.shape == (3, 200)
    assert np.array_equal(actual, expected)
    assert actual.T.tobytes() == expected.T.tobytes()


def test_sample_is_copy():
    data = _data()
    before = data.data.copy()
    rows, cols = _coords()
    data.sample(["nir", "blue"], rows, cols)[:] = -1
    assert np.array_equal(data.data, before)


class _RecordingModel:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.seen = []

    def predict_proba(self, matrix, n_jobs=0):
        self.seen.append(np.array(matrix))
        return np.tile([0.25, 0.75], (matrix.shape[0], 1))


@pytest.mark.parametrize("use_lat_lon", [False, True])
def test_classify_gives_model_the_original_matrix(use_lat_lon):
    data = _data()
    predictors = ["nir", "blue", "red"] + (["latitude", "longitude"] if use_lat_lon else [])
    obsmask = np.ones((30, 40), dtype=bool)
    obsmask[3:7] = False
    stub = SimpleNamespace(
        image=SimpleNamespace(
            data=data, obsmask=obsmask, shape=(30, 40), lat_center=41.5, lon_center=-72.25
        ),
        subsampling_size=2,
        subsampling_min=0,
        predictors=predictors,
        classes=["noncloud", "cloud"],
        model=_RecordingModel(),
    )
    LightGBM.classify(stub, probability="cloud")

    mask = np.zeros((30, 40), dtype=bool)
    mask[::2, ::2] = True
    rows, cols = np.where(mask & obsmask)
    expected = data.get(["nir", "blue", "red"])[:, rows, cols]
    if use_lat_lon:
        n_pixels = len(rows)
        expected = np.concatenate(
            (
                expected,
                np.full((1, n_pixels), 41.5, dtype=np.float32),
                np.full((1, n_pixels), -72.25, dtype=np.float32),
            ),
            axis=0,
        )
    seen = stub.model.seen[0]
    assert seen.dtype == expected.dtype
    assert seen.tobytes() == expected.T.tobytes()
