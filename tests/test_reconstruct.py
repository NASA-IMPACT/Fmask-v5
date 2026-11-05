import numpy as np
import pytest
from skimage.morphology import reconstruction

from fmask import _reconstruct
from fmask._reconstruct import reconstruct_erosion


def _hole_seed(mask):
    seed = mask.copy()
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        seed[1:-1, 1:-1] = np.nanmax(mask)
    return seed


def _random(dtype, seed=0, shape=(40, 57)):
    return np.random.default_rng(seed).random(shape).astype(dtype)


def _ties(dtype):
    return np.random.default_rng(1).integers(0, 4, (40, 57)).astype(dtype)


def _with_nan(dtype):
    mask = _random(dtype, 2)
    mask[5, 7] = np.nan
    mask[20:22, 30:33] = np.nan
    return mask


def _signed_zeros(dtype):
    rng = np.random.default_rng(3)
    mask = rng.integers(-1, 3, (40, 57)).astype(dtype)
    mask[mask == 0] = rng.choice([-0.0, 0.0], size=int((mask == 0).sum()))
    return mask


def _only_negative_zero(dtype):
    return np.full((9, 11), -0.0, dtype=dtype)


def _infinities(dtype):
    mask = _random(dtype, 4)
    mask[3, 3] = -np.inf
    mask[0, 4] = -np.inf
    return mask


def _negative_values(dtype):
    return (_random(dtype, 5) - 0.5).astype(dtype)


CASES = {
    "random": _random,
    "ties": _ties,
    "constant": lambda dtype: np.full((9, 11), 2.5, dtype=dtype),
    "nan": _with_nan,
    "signed_zeros": _signed_zeros,
    "only_negative_zero": _only_negative_zero,
    "infinities": _infinities,
    "negative_values": _negative_values,
    "one_by_one": lambda dtype: np.array([[1.5]], dtype=dtype),
    "one_by_n": lambda dtype: _random(dtype, 6, (1, 9)),
    "n_by_one": lambda dtype: _random(dtype, 7, (9, 1)),
}


def _assert_same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_skimage(case, dtype):
    mask = CASES[case](dtype)
    seed = _hole_seed(mask)
    _assert_same(
        reconstruct_erosion(seed, mask),
        reconstruction(seed, mask, method="erosion"),
    )


@pytest.mark.parametrize(
    "mask_dtype,seed_dtype",
    [
        (np.uint8, np.float32),
        (np.int16, np.float64),
        (np.bool_, np.float32),
        (np.float16, np.float64),
    ],
)
def test_matches_skimage_integer_and_small_types(mask_dtype, seed_dtype):
    mask = np.random.default_rng(8).integers(0, 2, (20, 23)).astype(mask_dtype)
    seed = _hole_seed(mask.astype(seed_dtype))
    _assert_same(
        reconstruct_erosion(seed, mask),
        reconstruction(seed, mask, method="erosion"),
    )


def test_seed_below_mask_raises_same_error():
    mask = np.ones((5, 5), dtype=np.float32)
    seed = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError) as expected:
        reconstruction(seed, mask, method="erosion")
    with pytest.raises(ValueError) as actual:
        reconstruct_erosion(seed, mask)
    assert str(actual.value) == str(expected.value)


def test_mirror_is_available():
    assert _reconstruct.MIRROR_AVAILABLE


@pytest.mark.parametrize(
    "version,expected",
    [("0.26.0", True), ("0.26.3", True), ("0.27.0", False), ("0.25.2", False), ("0.260.0", False)],
)
def test_mirror_is_limited_to_the_mirrored_version(version, expected):
    assert _reconstruct._mirror_matches(version) is expected


def test_uses_skimage_when_mirror_is_unavailable(monkeypatch):
    calls = []

    def spy(seed, mask, method):
        calls.append(method)
        return reconstruction(seed, mask, method=method)

    monkeypatch.setattr(_reconstruct, "MIRROR_AVAILABLE", False)
    monkeypatch.setattr(_reconstruct, "reconstruction", spy)
    mask = _random(np.float32, 13)
    seed = _hole_seed(mask)
    _assert_same(reconstruct_erosion(seed, mask), reconstruction(seed, mask, method="erosion"))
    assert calls == ["erosion"]
