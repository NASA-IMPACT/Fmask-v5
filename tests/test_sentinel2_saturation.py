import inspect

import numpy as np
import pandas
import pytest
import rasterio
from rasterio.transform import from_origin

from fmask.satellite import Landsat, Sentinel2

TARGET = 64
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


@pytest.fixture(params=[-1000.0, -1001.0, 0.0], ids=["offset_1000", "offset_1001", "no_offset"])
def scene(tmp_path, request):
    rng = np.random.default_rng(0)
    for _, filename, res in BANDS:
        size = TARGET * RES // res
        data = rng.integers(1000, 12000, (size, size), dtype=np.uint16)
        _write(tmp_path / f"{filename}.jp2", data, res, "JP2OpenJPEG")
    reference = tmp_path / "reference.tif"
    _write(reference, np.ones((TARGET, TARGET), dtype=np.uint16), RES, "GTiff")
    with rasterio.open(reference) as src:
        profile = src.meta.copy()
        profilefull = src
    image = object.__new__(Sentinel2)
    image.folder = str(tmp_path)
    image.bands = pandas.DataFrame(
        {
            "NAME": [name for name, _, _ in BANDS],
            "FILE": [filename for _, filename, _ in BANDS],
            "RADIO_ADD_OFFSET": [request.param] * len(BANDS),
            "QUANTIFICATION_VALUE": [QUANTIFICATION] * len(BANDS),
        }
    )
    image.profile = profile
    image.profilefull = profilefull
    image.obsmask = None
    image.nodata_index = 0.0
    image.saturation_index = 65535.0
    image.lat_north = 0.0
    image.lat_south = 0.0
    return image


def _set_dn(image, band, row, col, value):
    """Set one target pixel in the band file, covering every native pixel inside it."""
    _, filename, res = next(entry for entry in BANDS if entry[0] == band.upper())
    path = f"{image.folder}/{filename}.jp2"
    with rasterio.open(path) as src:
        data = src.read(1)
    scale = RES // res
    data[row * scale:(row + 1) * scale, col * scale:(col + 1) * scale] = value
    _write(path, data, res, "JP2OpenJPEG")


def test_read_band_arguments_match_landsat():
    assert inspect.signature(Sentinel2.read_band) == inspect.signature(Landsat.read_band)


@pytest.mark.parametrize("level", ["TOA", "DN"])
def test_read_band_returns_saturated_pixels(scene, level):
    _set_dn(scene, "blue", 50, 50, 65535)
    _set_dn(scene, "blue", 40, 40, 65534)

    _, satu = scene.read_band("blue", level=level, saturation=True)

    expected = np.zeros((TARGET, TARGET), dtype=bool)
    expected[50, 50] = True
    assert np.array_equal(satu, expected)


def test_read_band_leaves_saturation_out_by_default(scene):
    _set_dn(scene, "blue", 50, 50, 65535)

    _, satu = scene.read_band("blue")

    assert satu is None


def test_read_datacube_sets_saturation_bits_in_blue_green_red_order(scene):
    for band, (row, col) in PIXELS.items():
        _set_dn(scene, band, row, col, 65535)

    _, statu = scene.read_datacube(list(SPECTRAL))

    assert statu.index == 2
    for bit, band in enumerate(PIXELS):
        expected = np.zeros((TARGET, TARGET), dtype=bool)
        expected[PIXELS[band]] = True
        assert np.array_equal(statu.get(bit), expected)


def test_read_datacube_does_not_flag_the_dn_below_saturation(scene):
    for band in PIXELS:
        _set_dn(scene, band, 50, 50, 65534)

    _, statu = scene.read_datacube(list(SPECTRAL))

    assert not (statu.get(0) | statu.get(1) | statu.get(2)).any()


@pytest.mark.parametrize("missing", list(PIXELS))
def test_read_datacube_without_visible_band_raises(scene, missing):
    predictors = [band for band in SPECTRAL if band != missing]
    with pytest.raises(ValueError, match=missing):
        scene.read_datacube(predictors)


@pytest.mark.parametrize("band", list(PIXELS))
def test_read_datacube_zeroes_whiteness_of_saturated_pixels(scene, band):
    _set_dn(scene, band, 50, 50, 65535)

    data, _ = scene.read_datacube(SPECTRAL + ["whiteness"])

    expected_nonzero = np.ones((TARGET, TARGET), dtype=bool)
    expected_nonzero[50, 50] = False
    assert np.array_equal(data.data[data.bands.index("whiteness")] != 0, expected_nonzero)
