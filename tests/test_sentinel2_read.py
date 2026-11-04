import numpy as np
import pandas
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from fmask import constant as C
from fmask.satellite import Sentinel2

TARGET = 96
RES = 20
# (name, file, resolution): 20 m is equal to the target, 10 m is averaged, 60 m is nearest.
BANDS = [
    ("COASTAL", "B01", 60), ("BLUE", "B02", 10), ("GREEN", "B03", 10), ("RED", "B04", 10),
    ("VRE1", "B05", 20), ("VRE2", "B06", 20), ("VRE3", "B07", 20), ("WNIR", "B08", 10),
    ("NIR", "B8A", 20), ("WV", "B09", 60), ("CIRRUS", "B10", 60), ("SWIR1", "B11", 20),
    ("SWIR2", "B12", 20),
]
SPECTRAL = [
    "coastal", "blue", "green", "red", "vre1", "vre2", "vre3",
    "wnir", "nir", "wv", "swir1", "swir2", "cirrus",
]
OFFSETS = {name: -1000.0 - index for index, (name, _, _) in enumerate(BANDS)}
QUANTIFICATION = 10000.0


def _write(path, data, res, driver):
    # One resolution level means no overviews, so reads really use the requested resampling.
    options = {"QUALITY": 100, "REVERSIBLE": "YES", "RESOLUTIONS": 1} if driver == "JP2OpenJPEG" else {}
    with rasterio.open(
        path, "w", driver=driver, height=data.shape[0], width=data.shape[1], count=1,
        dtype="uint16", crs="EPSG:32633", transform=from_origin(500000, 4000000, res, res),
        **options,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def scene(tmp_path):
    rng = np.random.default_rng(0)
    for offset, (_, filename, res) in enumerate(BANDS):
        size = TARGET * RES // res
        data = rng.integers(1000, 12000, (size, size), dtype=np.uint16)
        data[offset, :] = 0
        data[:, offset] = 0
        _write(tmp_path / f"{filename}.jp2", data, res, "JP2OpenJPEG")
    reference = tmp_path / "reference.tif"
    _write(reference, np.ones((TARGET, TARGET), dtype=np.uint16), RES, "GTiff")
    with rasterio.open(reference) as src:
        profile = src.meta.copy()
        profilefull = src
    return {
        "folder": str(tmp_path),
        "bands": pandas.DataFrame(
            {
                "NAME": [name for name, _, _ in BANDS],
                "FILE": [filename for _, filename, _ in BANDS],
                "RADIO_ADD_OFFSET": [OFFSETS[name] for name, _, _ in BANDS],
                "QUANTIFICATION_VALUE": [QUANTIFICATION] * len(BANDS),
            }
        ),
        "profile": profile,
        "profilefull": profilefull,
    }


def _make(scene):
    image = object.__new__(Sentinel2)
    image.folder = scene["folder"]
    image.bands = scene["bands"]
    image.profile = scene["profile"]
    image.profilefull = scene["profilefull"]
    image.obsmask = None
    image.nodata_index = 0.0
    image.saturation_index = 65535.0
    image.lat_north = 0.0
    image.lat_south = 0.0
    return image


def _assert_same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def _raw(scene, band):
    name, filename, res = next(entry for entry in BANDS if entry[0] == band.upper())
    with rasterio.open(f"{scene['folder']}/{filename}.jp2") as src:
        if res == RES:
            return src.read(1)
        resampling = Resampling.average if res < RES else Resampling.nearest
        return src.read(1, out_shape=(TARGET, TARGET), resampling=resampling)


def _expected_toa(scene, band):
    return (_raw(scene, band) + OFFSETS[band.upper()]) / QUANTIFICATION


def _expected_obsmask(scene, bands):
    return np.logical_and.reduce([_raw(scene, band) != 0 for band in bands])


@pytest.mark.parametrize("band", ["vre1", "blue", "coastal"], ids=["equal", "average", "nearest"])
def test_read_band_converts_to_toa(scene, band):
    result, satu = _make(scene).read_band(band)
    _assert_same(result, _expected_toa(scene, band))
    assert satu is None


@pytest.mark.parametrize("band", ["vre1", "blue", "coastal"], ids=["equal", "average", "nearest"])
def test_read_band_returns_dn_for_other_levels(scene, band):
    result, _ = _make(scene).read_band(band, level="DN")
    _assert_same(result, _raw(scene, band))
    assert result.shape == (TARGET, TARGET)


def test_read_band_overlaps_the_observation_mask(scene):
    image = _make(scene)
    image.read_band("vre1")
    _assert_same(image.obsmask, _raw(scene, "vre1") != 0)
    image.read_band("blue")
    _assert_same(image.obsmask, _expected_obsmask(scene, ["vre1", "blue"]))


def test_read_band_can_leave_the_observation_mask(scene):
    image = _make(scene)
    image.read_band("blue", maskupdate=False)
    assert image.obsmask is None


def test_read_band_profile(scene):
    image = _make(scene)
    image.read_band("vre1", profile=True)
    with rasterio.open(f"{scene['folder']}/B05.jp2") as src:
        assert image.profile == src.meta
    assert image.profilefull.name.endswith("B05.jp2")
    assert image.profilefull.closed


def test_values_do_not_touch_self(scene):
    image = _make(scene)
    mask = np.ones((TARGET, TARGET), dtype=bool)
    profile, profilefull = image.profile, image.profilefull
    image.obsmask = mask
    _, _, valid = image._read_band_values("vre1", "TOA")
    assert image.obsmask is mask
    assert image.profile is profile
    assert image.profilefull is profilefull
    assert valid.dtype == np.bool_


def test_obsmask_all_bands(scene):
    image = _make(scene)
    image.read_datacube(list(SPECTRAL))
    _assert_same(image.obsmask, _expected_obsmask(scene, SPECTRAL))
    assert not image.obsmask.all()
    assert image.obsmask.any()


@pytest.mark.parametrize("maskupdate", [True, False])
@pytest.mark.parametrize("predictors", [SPECTRAL, list(reversed(SPECTRAL))], ids=["forward", "reversed"])
def test_read_datacube_matches_each_band(scene, predictors, maskupdate):
    image = _make(scene)
    data, _ = image.read_datacube(list(predictors), maskupdate=maskupdate)
    expected = np.stack([_expected_toa(scene, band) for band in predictors]).astype(np.float32)
    if maskupdate:
        expected[:, ~_expected_obsmask(scene, SPECTRAL)] = 0
    else:
        assert image.obsmask is None
    _assert_same(data.data, expected)
    assert data.bands == list(predictors)


def test_read_datacube_replaces_nan_values(scene, monkeypatch):
    original = Sentinel2._read_band_values

    def with_nan(self, band, **kwargs):
        image, satu, valid = original(self, band, **kwargs)
        image[20, 20] = np.nan
        return image, satu, valid

    monkeypatch.setattr(Sentinel2, "_read_band_values", with_nan)
    data, _ = _make(scene).read_datacube(list(SPECTRAL))
    assert (data.data[:, 20, 20] == np.float32(C.EPS)).all()
