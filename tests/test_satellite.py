import gc
import inspect
import threading
import time
import weakref

import numpy as np
import pytest

from fmask import satellite
from fmask.satellite import Data, Landsat, Satellite, Sentinel2

BANDS = ["blue", "green", "red", "nir"]


class _Reader(Satellite):
    def _read_band_values(self, band, **kwargs):
        raise NotImplementedError


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


def test_read_bands_order(monkeypatch):
    image = object.__new__(_Reader)
    release = threading.Event()

    def fake_values(self, band, **kwargs):
        if band == "first":
            assert release.wait(timeout=10)
        else:
            release.set()
        return band

    monkeypatch.setattr(_Reader, "_read_band_values", fake_values)
    requests = [{"band": "first", "level": "TOA"}, {"band": "second", "level": "TOA"}]
    assert list(image._read_bands(requests)) == ["first", "second"]


def test_read_bands_limits_threads(monkeypatch):
    image = object.__new__(_Reader)
    lock = threading.Lock()
    active = [0, 0]
    barrier = threading.Barrier(satellite.READ_THREADS)

    def fake_values(self, band, **kwargs):
        with lock:
            active[0] += 1
            active[1] = max(active)
        barrier.wait(timeout=10)
        with lock:
            active[0] -= 1
        return band

    monkeypatch.setattr(_Reader, "_read_band_values", fake_values)
    count = satellite.READ_THREADS * 3
    list(image._read_bands([{"band": str(i), "level": "TOA"} for i in range(count)]))
    assert active[1] == satellite.READ_THREADS


def test_read_bands_propagates_errors(monkeypatch):
    image = object.__new__(_Reader)

    def fake_values(self, band, **kwargs):
        raise OSError("bad band")

    monkeypatch.setattr(_Reader, "_read_band_values", fake_values)
    with pytest.raises(OSError, match="bad band"):
        list(image._read_bands([{"band": "blue", "level": "TOA"}]))


def test_worker_threads_use_caller_error_state(monkeypatch):
    image = object.__new__(_Reader)
    monkeypatch.setattr(_Reader, "_read_band_values", lambda self, band, **kw: np.geterr()["invalid"])
    with np.errstate(invalid="ignore"):
        assert list(image._read_bands([{"band": "blue", "level": "TOA"}])) == ["ignore"]
    with np.errstate(invalid="raise"):
        assert list(image._read_bands([{"band": "blue", "level": "TOA"}])) == ["raise"]


def test_read_bands_releases_each_result_before_the_next(monkeypatch):
    image = object.__new__(_Reader)

    class Value:
        pass

    refs = {}

    def fake_values(self, band, **kwargs):
        value = Value()
        refs[band] = weakref.ref(value)
        return value

    monkeypatch.setattr(_Reader, "_read_band_values", fake_values)
    results = image._read_bands([{"band": str(i), "level": "TOA"} for i in range(3)])
    next(results)
    gc.collect()
    value = next(results)
    gc.collect()
    assert refs["0"]() is None
    del value
    list(results)


def test_read_bands_limits_results_waiting_behind_a_slow_band(monkeypatch):
    image = object.__new__(_Reader)
    started = []
    started_during_slow_band = []

    def fake_values(self, band, **kwargs):
        started.append(band)
        if band == "0":
            time.sleep(0.3)
            started_during_slow_band.append(len(started))
        return band

    monkeypatch.setattr(_Reader, "_read_band_values", fake_values)
    count = satellite.READ_THREADS * 3
    requests = [{"band": str(i), "level": "TOA"} for i in range(count)]
    assert list(image._read_bands(requests)) == [str(i) for i in range(count)]
    assert started_during_slow_band == [satellite.READ_THREADS]


def test_read_band_arguments_match_landsat():
    assert inspect.signature(Sentinel2.read_band) == inspect.signature(Landsat.read_band)
