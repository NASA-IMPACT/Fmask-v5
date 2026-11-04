import gc
import sys
import threading
import time
import weakref

import numpy as np
import pandas
import pytest
import rasterio
from rasterio.transform import from_origin

from fmask import constant as C
from fmask import satellite, utils
from fmask.satellite import Landsat

SHAPE = (48, 64)
BAND_IDS = {
    "COASTAL": "1", "BLUE": "2", "GREEN": "3", "RED": "4", "NIR": "5",
    "SWIR1": "6", "SWIR2": "7", "CIRRUS": "9", "TIRS1": "10", "TIRS2": "11",
}
SPECTRAL = ["coastal", "blue", "green", "red", "nir", "swir1", "swir2", "tirs1", "tirs2", "cirrus"]


def _write(path, data, dtype, nodata):
    with rasterio.open(
        path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1, dtype=dtype,
        crs="EPSG:32633", transform=from_origin(500000, 4000000, 30, 30), nodata=nodata,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def scene(tmp_path):
    rng = np.random.default_rng(0)
    record, radiometric, thermal = {}, {}, {}
    for offset, (name, band_id) in enumerate(BAND_IDS.items()):
        data = rng.integers(5000, 50000, SHAPE, dtype=np.uint16)
        data[offset, :] = 0
        data[:, offset] = 0
        data[-1, -1 - offset] = np.iinfo(np.uint16).max
        filename = f"band{band_id}.tif"
        _write(tmp_path / filename, data, "uint16", 0)
        record[f"FILE_NAME_BAND_{band_id}"] = filename
        radiometric[f"REFLECTANCE_MULT_BAND_{band_id}"] = str(2.0e-5 + offset * 1e-7)
        radiometric[f"REFLECTANCE_ADD_BAND_{band_id}"] = str(-0.1 - offset * 1e-3)
        radiometric[f"RADIANCE_MULT_BAND_{band_id}"] = str(3.342e-4)
        radiometric[f"RADIANCE_ADD_BAND_{band_id}"] = str(0.1 + offset * 1e-2)
        thermal[f"K1_CONSTANT_BAND_{band_id}"] = str(774.8853 + offset)
        thermal[f"K2_CONSTANT_BAND_{band_id}"] = str(1321.0789 + offset)
    angle = rng.integers(2000, 7000, SHAPE, dtype=np.int16)
    _write(tmp_path / "sza.tif", angle, "int16", None)
    record["FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4"] = "sza.tif"
    return {
        "folder": str(tmp_path),
        "metadata": {
            "LEVEL1_PROCESSING_RECORD": record,
            "LEVEL1_RADIOMETRIC_RESCALING": radiometric,
            "LEVEL1_THERMAL_CONSTANTS": thermal,
        },
        "bands": pandas.DataFrame(
            {"NAME": list(BAND_IDS), "ID": list(BAND_IDS.values())}
        ),
    }


def _make(cls, scene):
    image = object.__new__(cls)
    image.folder = scene["folder"]
    image.metadata = scene["metadata"]
    image.bands = scene["bands"]
    image.obsmask = None
    image.profile = None
    image.profilefull = None
    return image


def _assert_same(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_values_do_not_touch_self(scene):
    image = _make(Landsat, scene)
    sentinel_mask = np.ones(SHAPE, dtype=bool)
    image.obsmask = sentinel_mask
    image.profile = "profile"
    image.profilefull = "profilefull"
    _, _, valid = image._read_band_values("blue", "TOA")
    assert image.obsmask is sentinel_mask
    assert image.profile == "profile"
    assert image.profilefull == "profilefull"
    assert valid.dtype == np.bool_
    assert not valid[1, :].any()


def test_read_bands_order(scene, monkeypatch):
    image = _make(Landsat, scene)
    release = threading.Event()

    def fake_values(self, band, **kwargs):
        if band == "first":
            assert release.wait(timeout=10)
        else:
            release.set()
        return band

    monkeypatch.setattr(Landsat, "_read_band_values", fake_values)
    requests = [{"band": "first", "level": "TOA"}, {"band": "second", "level": "TOA"}]
    assert list(image._read_bands(requests)) == ["first", "second"]


def test_read_bands_limits_threads(scene, monkeypatch):
    image = _make(Landsat, scene)
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

    monkeypatch.setattr(Landsat, "_read_band_values", fake_values)
    count = satellite.READ_THREADS * 3
    list(image._read_bands([{"band": str(i), "level": "TOA"} for i in range(count)]))
    assert active[1] == satellite.READ_THREADS


def test_read_bands_propagates_errors(scene, monkeypatch):
    image = _make(Landsat, scene)

    def fake_values(self, band, **kwargs):
        raise OSError("bad band")

    monkeypatch.setattr(Landsat, "_read_band_values", fake_values)
    with pytest.raises(OSError, match="bad band"):
        list(image._read_bands([{"band": "blue", "level": "TOA"}]))


def test_worker_threads_use_caller_error_state(scene, monkeypatch):
    image = _make(Landsat, scene)
    monkeypatch.setattr(Landsat, "_read_band_values", lambda self, band, **kw: np.geterr()["invalid"])
    with np.errstate(invalid="ignore"):
        assert list(image._read_bands([{"band": "blue", "level": "TOA"}])) == ["ignore"]
    with np.errstate(invalid="raise"):
        assert list(image._read_bands([{"band": "blue", "level": "TOA"}])) == ["raise"]


def test_read_bands_releases_each_result_before_the_next(scene, monkeypatch):
    image = _make(Landsat, scene)

    class Value:
        pass

    refs = {}

    def fake_values(self, band, **kwargs):
        value = Value()
        refs[band] = weakref.ref(value)
        return value

    monkeypatch.setattr(Landsat, "_read_band_values", fake_values)
    results = image._read_bands([{"band": str(i), "level": "TOA"} for i in range(3)])
    next(results)
    gc.collect()
    value = next(results)
    gc.collect()
    assert refs["0"]() is None
    del value
    list(results)


def test_read_bands_limits_results_waiting_behind_a_slow_band(scene, monkeypatch):
    image = _make(Landsat, scene)
    started = []
    started_during_slow_band = []

    def fake_values(self, band, **kwargs):
        started.append(band)
        if band == "0":
            time.sleep(0.3)
            started_during_slow_band.append(len(started))
        return band

    monkeypatch.setattr(Landsat, "_read_band_values", fake_values)
    count = satellite.READ_THREADS * 3
    requests = [{"band": str(i), "level": "TOA"} for i in range(count)]
    assert list(image._read_bands(requests)) == [str(i) for i in range(count)]
    assert started_during_slow_band == [satellite.READ_THREADS]


def test_read_datacube_releases_read_inputs_before_indices(scene, monkeypatch):
    original_hot = satellite.hot
    seen = {}

    def spy(*args):
        seen.update(sys._getframe(1).f_locals)
        return original_hot(*args)

    monkeypatch.setattr(satellite, "hot", spy)
    _make(Landsat, scene).read_datacube(SPECTRAL + ["hot"])
    assert seen
    assert not {"request", "requests", "cos_sza", "saturated", "image", "satu", "valid"} & set(seen)


def _raw(scene, band):
    filename = scene["metadata"]["LEVEL1_PROCESSING_RECORD"][f"FILE_NAME_BAND_{BAND_IDS[band.upper()]}"]
    with rasterio.open(f"{scene['folder']}/{filename}") as src:
        return src.read(1)


def _cos_sza(scene):
    with rasterio.open(f"{scene['folder']}/sza.tif") as src:
        return np.cos(np.deg2rad(src.read(1).astype(np.float32) / 100))


def _expected_toa(scene, band):
    band_id = BAND_IDS[band.upper()]
    rescaling = scene["metadata"]["LEVEL1_RADIOMETRIC_RESCALING"]
    image = _raw(scene, band).astype(np.float32)
    image = image * float(rescaling[f"REFLECTANCE_MULT_BAND_{band_id}"])
    image = image + float(rescaling[f"REFLECTANCE_ADD_BAND_{band_id}"])
    return image / _cos_sza(scene)


def _expected_bt(scene, band):
    band_id = BAND_IDS[band.upper()]
    rescaling = scene["metadata"]["LEVEL1_RADIOMETRIC_RESCALING"]
    thermal = scene["metadata"]["LEVEL1_THERMAL_CONSTANTS"]
    image = _raw(scene, band).astype(np.float32)
    image = image * float(rescaling[f"RADIANCE_MULT_BAND_{band_id}"])
    image = image + float(rescaling[f"RADIANCE_ADD_BAND_{band_id}"])
    image = image + C.EPS
    image = float(thermal[f"K2_CONSTANT_BAND_{band_id}"]) / np.log(
        float(thermal[f"K1_CONSTANT_BAND_{band_id}"]) / image + 1
    )
    return image - 273.15


def _expected_band(scene, band):
    return _expected_bt(scene, band) if band.startswith("tirs") else _expected_toa(scene, band)


def _expected_obsmask(scene, bands):
    return np.logical_and.reduce([_raw(scene, band) != 0 for band in bands])


@pytest.mark.parametrize("pass_sza", [True, False], ids=["sza_passed", "sza_read"])
def test_read_band_converts_to_toa(scene, pass_sza):
    image = _make(Landsat, scene)
    sza = image.read_angle(angle="SOLAR_ZENITH", unit="RADIAN") if pass_sza else None
    result, satu = image.read_band("red", level="TOA", sza=sza)
    _assert_same(result, _expected_toa(scene, "red"))
    assert satu is None


def test_read_band_converts_to_brightness_temperature(scene):
    result, satu = _make(Landsat, scene).read_band("tirs1", level="BT")
    _assert_same(result, _expected_bt(scene, "tirs1"))
    assert satu is None


def test_read_band_returns_saturated_pixels(scene):
    _, satu = _make(Landsat, scene).read_band("green", saturation=True)
    _assert_same(satu, _raw(scene, "green") == np.iinfo(np.uint16).max)
    assert satu.any()


def test_read_band_overlaps_the_observation_mask(scene):
    image = _make(Landsat, scene)
    image.read_band("coastal")
    _assert_same(image.obsmask, _raw(scene, "coastal") != 0)
    image.read_band("blue")
    _assert_same(image.obsmask, _expected_obsmask(scene, ["coastal", "blue"]))


def test_read_band_can_leave_the_observation_mask(scene):
    image = _make(Landsat, scene)
    image.read_band("blue", maskupdate=False)
    assert image.obsmask is None


def test_read_band_profile(scene):
    image = _make(Landsat, scene)
    image.read_band("blue", profile=True)
    with rasterio.open(f"{scene['folder']}/band2.tif") as src:
        assert image.profile == src.meta
    assert image.profilefull.name.endswith("band2.tif")
    assert image.profilefull.closed


def test_obsmask_all_bands(scene):
    image = _make(Landsat, scene)
    image.read_datacube(list(SPECTRAL))
    _assert_same(image.obsmask, _expected_obsmask(scene, SPECTRAL))
    assert not image.obsmask.all()
    assert image.obsmask.any()


@pytest.mark.parametrize("maskupdate", [True, False])
@pytest.mark.parametrize("predictors", [SPECTRAL, list(reversed(SPECTRAL))], ids=["forward", "reversed"])
def test_read_datacube_matches_each_band(scene, predictors, maskupdate):
    image = _make(Landsat, scene)
    data, _ = image.read_datacube(list(predictors), maskupdate=maskupdate)
    expected = np.stack([_expected_band(scene, band) for band in predictors])
    if maskupdate:
        expected[:, ~_expected_obsmask(scene, SPECTRAL)] = 0
    else:
        assert image.obsmask is None
    _assert_same(data.data, expected)
    assert data.bands == list(predictors)


def test_read_datacube_sets_saturation_bits_in_blue_green_red_order(scene):
    _, statu = _make(Landsat, scene).read_datacube(list(SPECTRAL))
    assert statu.index == 2
    for bit, band in enumerate(["blue", "green", "red"]):
        _assert_same(statu.get(bit), _raw(scene, band) == np.iinfo(np.uint16).max)
    assert not np.array_equal(statu.get(0), statu.get(1))


def test_read_datacube_computes_indices_from_the_bands_read(scene):
    data, _ = _make(Landsat, scene).read_datacube(SPECTRAL + ["hot", "ndvi"])
    mask = _expected_obsmask(scene, SPECTRAL)
    blue, red, nir = (_expected_toa(scene, band) for band in ("blue", "red", "nir"))
    for name, expected in (("hot", utils.hot(blue, red)), ("ndvi", utils.ndvi(red, nir))):
        expected = expected.astype(np.float32).copy()
        expected[~mask] = 0
        _assert_same(data.data[data.bands.index(name)], expected)


def test_read_datacube_replaces_nan_values(scene, monkeypatch):
    original = Landsat._read_band_values

    def with_nan(self, band, **kwargs):
        image, satu, valid = original(self, band, **kwargs)
        image[20, 20] = np.nan
        return image, satu, valid

    monkeypatch.setattr(Landsat, "_read_band_values", with_nan)
    data, _ = _make(Landsat, scene).read_datacube(list(SPECTRAL))
    assert (data.data[:, 20, 20] == np.float32(C.EPS)).all()


@pytest.mark.parametrize("missing", ["blue", "green", "red"])
def test_read_datacube_without_visible_band_raises(scene, missing):
    predictors = [band for band in SPECTRAL if band != missing]
    with pytest.raises(ValueError, match=missing):
        _make(Landsat, scene).read_datacube(predictors)
