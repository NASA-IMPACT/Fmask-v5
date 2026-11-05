import numpy as np
import pytest
from skimage.morphology import reconstruction

from fmask import constant as C
from fmask import utils
from fmask.satellite import Data

SHAPE = (4, 23, 31)


def _cube(seed=0, dtype=np.float32):
    return (np.random.default_rng(seed).random(SHAPE) * 3000 - 500).astype(dtype)


def _partial_mask():
    mask = np.ones(SHAPE[1:], dtype=bool)
    mask[:4] = False
    mask[:, -3:] = False
    mask[10, 10] = False
    return mask


def _assert_same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def _bands(cube, **kwargs):
    return utils.normalize_bands(cube, list(range(cube.shape[0])), **kwargs)


def _scale_in_float64(band, valid, percentiles=(1, 99), srange=(-1, 1)):
    low, high = np.percentile(valid.astype("float"), percentiles, method="linear")
    scaled = ((band.astype("float") - low) / (high - low + C.EPS)) * (srange[1] - srange[0]) + srange[0]
    return np.clip(scaled, srange[0], srange[1])


def test_known_values():
    cube = np.array([[[0.0, 25.0, 50.0, 75.0, 100.0]]], dtype=np.float32)
    result = _bands(cube, percentiles=(0, 100))
    expected = (np.array([0.0, 25.0, 50.0, 75.0, 100.0]) / (100.0 + C.EPS)) * 2 - 1
    _assert_same(result, expected.astype(np.float32).reshape(1, 1, 5))


def test_scales_in_float64_before_casting():
    cube = _cube(1)
    expected = np.stack(
        [_scale_in_float64(band, band).astype(np.float32) for band in cube]
    )
    _assert_same(_bands(cube), expected)

    low, high = np.percentile(cube[0].astype("float"), (1, 99), method="linear")
    in_float32 = ((cube[0] - np.float32(low)) / np.float32(high - low + C.EPS)) * 2 - 1
    assert in_float32.tobytes() != expected[0].tobytes()


def test_percentiles_ignore_pixels_outside_the_mask():
    cube, mask = _cube(2), _partial_mask()
    with_outliers = cube.copy()
    with_outliers[:, ~mask] = 1e9
    _assert_same(_bands(with_outliers, obsmask=mask), _bands(cube, obsmask=mask))


@pytest.mark.parametrize("srange", [(-1, 1), (0, 1), (-3, 7)])
def test_pixels_outside_the_mask_get_the_range_minimum(srange):
    result = _bands(_cube(3), obsmask=_partial_mask(), srange=srange)
    assert (result[:, ~_partial_mask()] == min(srange)).all()


@pytest.mark.parametrize("percentiles,srange", [((1, 99), (-1, 1)), ((5, 50), (-3, 7))])
def test_values_inside_the_mask_stay_in_range(percentiles, srange):
    mask = _partial_mask()
    result = _bands(_cube(4), obsmask=mask, percentiles=percentiles, srange=srange)
    assert result.min() >= srange[0]
    assert result.max() <= srange[1]
    assert result[:, mask].min() == srange[0]
    assert result[:, mask].max() == srange[1]


def test_constant_band_gets_the_range_minimum():
    cube = _cube(5)
    cube[0] = 17.5
    assert (_bands(cube)[0] == -1).all()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nan_inside_the_mask_only_affects_its_band():
    cube, mask = _cube(6), _partial_mask()
    clean = _bands(cube, obsmask=mask)
    cube[1, 12, 12] = np.nan
    result = _bands(cube, obsmask=mask)
    assert np.isnan(result[1, mask]).all()
    assert (result[1, ~mask] == -1).all()
    _assert_same(result[[0, 2, 3]], clean[[0, 2, 3]])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nan_outside_the_mask_is_replaced():
    cube, mask = _cube(7), _partial_mask()
    clean = _bands(cube, obsmask=mask)
    cube[2, ~mask] = np.nan
    _assert_same(_bands(cube, obsmask=mask), clean)


@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("block_rows", [1, 7, 1000])
def test_block_size_does_not_change_results(block_rows, use_mask):
    cube = _cube(8)
    obsmask = _partial_mask() if use_mask else None
    _assert_same(
        _bands(cube, obsmask=obsmask, block_rows=block_rows),
        _bands(cube, obsmask=obsmask, block_rows=SHAPE[1]),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_result_keeps_the_input_dtype(dtype):
    assert _bands(_cube(9, dtype)).dtype == dtype


def test_normalize_datacube_matches_normalize_bands():
    cube, mask = _cube(10), _partial_mask()
    _assert_same(utils.normalize_datacube(cube, obsmask=mask), _bands(cube, obsmask=mask))
    _assert_same(utils.normalize_datacube(cube), _bands(cube))


def test_empty_obsmask_raises():
    with pytest.raises(IndexError):
        _bands(_cube(11), obsmask=np.zeros(SHAPE[1:], dtype=bool))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_input_not_mutated():
    mask = _partial_mask()
    cube = _cube(12)
    cube[1, 12, 12] = np.nan
    cube_before, mask_before = cube.copy(), mask.copy()
    _bands(cube, obsmask=mask)
    _assert_same(cube, cube_before)
    _assert_same(mask, mask_before)


def test_result_does_not_share_memory_with_input():
    cube = _cube(13)
    assert not np.shares_memory(_bands(cube), cube)


def test_bands_subset_follows_the_requested_order():
    cube, mask = _cube(14), _partial_mask()
    _assert_same(
        utils.normalize_bands(cube, [2, 0], obsmask=mask),
        _bands(cube[[2, 0]], obsmask=mask),
    )


def test_predictor_order_matches_data_get():
    cube, mask = _cube(15), _partial_mask()
    data = Data(cube, ["blue", "green", "red", "nir"])
    predictors = ["nir", "blue", "red"]
    _assert_same(
        utils.normalize_bands(data.data, [data.bands.index(p) for p in predictors], obsmask=mask),
        _bands(data.get(predictors), obsmask=mask),
    )


def _random(dtype, seed=0, shape=(40, 57)):
    return np.random.default_rng(seed).random(shape).astype(dtype)


@pytest.mark.parametrize("fill_value", [None, -1.0])
@pytest.mark.parametrize("obs_shape", ["full", "partial", "single"])
def test_imfill_matches_skimage(obs_shape, fill_value):
    data = _random(np.float32, 11, (31, 37)) * 10
    obsmask = np.ones(data.shape, dtype=bool)
    if obs_shape == "partial":
        obsmask[:5] = False
        obsmask[:, -4:] = False
    elif obs_shape == "single":
        obsmask[:] = False
        obsmask[15, 15] = True
    max_val = data[obsmask].max()
    filled_input = data.copy()
    filled_input[~obsmask] = max_val if fill_value is None else fill_value
    seed = filled_input.copy()
    seed[1:-1, 1:-1] = max_val

    result = utils.imfill(data, obsmask, fill_value=fill_value)

    _assert_same(result, reconstruction(seed, filled_input, method="erosion"))
    _assert_same(data, filled_input)


def test_imfill_fills_a_closed_hole():
    data = np.full((5, 5), 5.0, dtype=np.float32)
    data[2, 2] = 1.0
    result = utils.imfill(data, np.ones((5, 5), dtype=bool))
    assert result[2, 2] == 5.0


def test_imfill_keeps_a_depression_open_to_the_border():
    data = np.full((5, 5), 5.0, dtype=np.float32)
    data[2, :3] = 1.0
    result = utils.imfill(data, np.ones((5, 5), dtype=bool))
    assert (result[2, :3] == 1.0).all()


def test_imfill_empty_observation_mask_raises():
    data = _random(np.float32, 12, (8, 8))
    with pytest.raises(ValueError):
        utils.imfill(data, np.zeros(data.shape, dtype=bool))
