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
PIXELS = {"blue": (50, 50), "green": (40, 40), "red": (30, 30)}
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


def _write_scene(folder, offsets, nodata_lines):
    rng = np.random.default_rng(0)
    for offset, (_, filename, res) in enumerate(BANDS):
        size = TARGET * RES // res
        data = rng.integers(1000, 12000, (size, size), dtype=np.uint16)
        if nodata_lines:
            data[offset, :] = 0
            data[:, offset] = 0
        _write(folder / f"{filename}.jp2", data, res, "JP2OpenJPEG")
    reference = folder / "reference.tif"
    _write(reference, np.ones((TARGET, TARGET), dtype=np.uint16), RES, "GTiff")
    with rasterio.open(reference) as src:
        profile = src.meta.copy()
        profilefull = src
    return {
        "folder": str(folder),
        "bands": pandas.DataFrame(
            {
                "NAME": [name for name, _, _ in BANDS],
                "FILE": [filename for _, filename, _ in BANDS],
                "RADIO_ADD_OFFSET": [offsets[name] for name, _, _ in BANDS],
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


@pytest.fixture
def scene(tmp_path):
    return _write_scene(tmp_path, OFFSETS, nodata_lines=True)


@pytest.fixture(params=[-1000.0, -1001.0, 0.0], ids=["offset_1000", "offset_1001", "no_offset"])
def clear_image(tmp_path, request):
    offsets = {name: request.param for name, _, _ in BANDS}
    return _make(_write_scene(tmp_path, offsets, nodata_lines=False))


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


def _set_dn(image, band, row, col, value):
    """Set one target pixel in the band file, covering every native pixel inside it."""
    _, filename, res = next(entry for entry in BANDS if entry[0] == band.upper())
    path = f"{image.folder}/{filename}.jp2"
    with rasterio.open(path) as src:
        data = src.read(1)
    scale = RES // res
    data[row * scale:(row + 1) * scale, col * scale:(col + 1) * scale] = value
    _write(path, data, res, "JP2OpenJPEG")


@pytest.mark.parametrize("level", ["TOA", "DN"])
def test_read_band_returns_saturated_pixels(clear_image, level):
    _set_dn(clear_image, "blue", 50, 50, 65535)
    _set_dn(clear_image, "blue", 40, 40, 65534)

    _, satu = clear_image.read_band("blue", level=level, saturation=True)

    expected = np.zeros((TARGET, TARGET), dtype=bool)
    expected[50, 50] = True
    assert np.array_equal(satu, expected)


def test_read_band_leaves_saturation_out_by_default(clear_image):
    _set_dn(clear_image, "blue", 50, 50, 65535)

    _, satu = clear_image.read_band("blue")

    assert satu is None


def test_read_datacube_sets_saturation_bits_in_blue_green_red_order(clear_image):
    for band, (row, col) in PIXELS.items():
        _set_dn(clear_image, band, row, col, 65535)

    _, statu = clear_image.read_datacube(list(SPECTRAL))

    assert statu.index == 2
    for bit, band in enumerate(PIXELS):
        expected = np.zeros((TARGET, TARGET), dtype=bool)
        expected[PIXELS[band]] = True
        assert np.array_equal(statu.get(bit), expected)


def test_read_datacube_does_not_flag_the_dn_below_saturation(clear_image):
    for band in PIXELS:
        _set_dn(clear_image, band, 50, 50, 65534)

    _, statu = clear_image.read_datacube(list(SPECTRAL))

    assert not (statu.get(0) | statu.get(1) | statu.get(2)).any()


@pytest.mark.parametrize("missing", list(PIXELS))
def test_read_datacube_without_visible_band_raises(clear_image, missing):
    predictors = [band for band in SPECTRAL if band != missing]
    with pytest.raises(ValueError, match=missing):
        clear_image.read_datacube(predictors)


@pytest.mark.parametrize("band", list(PIXELS))
def test_read_datacube_zeroes_whiteness_of_saturated_pixels(clear_image, band):
    _set_dn(clear_image, band, 50, 50, 65535)

    data, _ = clear_image.read_datacube(SPECTRAL + ["whiteness"])

    expected_nonzero = np.ones((TARGET, TARGET), dtype=bool)
    expected_nonzero[50, 50] = False
    assert np.array_equal(data.data[data.bands.index("whiteness")] != 0, expected_nonzero)
