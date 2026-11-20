"""Grayscale reconstruction by erosion that sorts the image once."""

import warnings

import numpy as np
import skimage
from skimage.morphology import reconstruction

MIRRORED_VERSION_PREFIX = "0.26."


def _mirror_matches(version: str) -> bool:
    """Return True for the scikit-image versions whose private internals this module mirrors."""
    return version.startswith(MIRRORED_VERSION_PREFIX)


try:
    from skimage._shared.utils import _supported_float_type
    from skimage.morphology._grayreconstruct import reconstruction_loop

    MIRROR_AVAILABLE = _mirror_matches(skimage.__version__)
except ImportError:
    MIRROR_AVAILABLE = False

if not MIRROR_AVAILABLE:
    warnings.warn(
        f"fmask._reconstruct mirrors scikit-image {MIRRORED_VERSION_PREFIX}x internals, so "
        f"scikit-image {skimage.__version__} uses its slower reconstruction, which gives the "
        "same results. Update fmask._reconstruct to regain the speedup.",
        RuntimeWarning,
        stacklevel=2,
    )


def _needs_skimage(images: np.ndarray) -> bool:
    """Return True for values whose result depends on which equal value skimage's sort keeps."""
    return bool(np.isnan(images).any() or np.signbit(images[images == 0]).any())


def reconstruct_erosion(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return skimage.morphology.reconstruction(seed, mask, method="erosion") for 2D images."""
    if not MIRROR_AVAILABLE:
        return reconstruction(seed, mask, method="erosion")
    assert tuple(seed.shape) == tuple(mask.shape)
    if np.any(seed < mask):
        raise ValueError(
            "Intensity of seed image must be greater than that "
            "of the mask image for reconstruction by erosion."
        )
    rows, cols = seed.shape
    padded_cols = cols + 2
    image_stride = (rows + 2) * padded_cols

    images = np.full(
        (2, rows + 2, padded_cols), np.max(seed), dtype=_supported_float_type(mask.dtype)
    )
    images[0, 1:-1, 1:-1] = seed
    images[1, 1:-1, 1:-1] = mask
    if _needs_skimage(images):
        return reconstruction(seed, mask, method="erosion")
    images = images.reshape(-1)

    isize = images.size
    signed_int_dtype = np.result_type(np.min_scalar_type(-isize), np.int32)
    unsigned_int_dtype = np.dtype(signed_int_dtype.char.upper())
    neighbor_strides = np.array(
        [
            -padded_cols - 1, -padded_cols, -padded_cols + 1,
            -1, 1,
            padded_cols - 1, padded_cols, padded_cols + 1,
        ],
        signed_int_dtype,
    )

    index_sorted = np.argsort(images).astype(signed_int_dtype, copy=False)
    sorted_values = images[index_sorted]
    del images
    is_different = sorted_values[:-1] != sorted_values[1:]

    ascending_rank = np.zeros(isize, unsigned_int_dtype)
    np.cumsum(is_different, out=ascending_rank[1:], dtype=unsigned_int_dtype)
    value_map = np.concatenate((sorted_values[:1], sorted_values[1:][is_different]))[::-1]
    del sorted_values, is_different

    # The reconstruction loop visits ranks in ascending order, so erosion needs the
    # largest value at rank 0.
    value_rank = np.empty(isize, unsigned_int_dtype)
    value_rank[index_sorted] = ascending_rank[-1] - ascending_rank
    del ascending_rank

    prev = np.full(isize, -1, signed_int_dtype)
    next_ = np.full(isize, -1, signed_int_dtype)
    prev[index_sorted[1:]] = index_sorted[:-1]
    next_[index_sorted[:-1]] = index_sorted[1:]
    start = index_sorted[0]
    del index_sorted

    reconstruction_loop(value_rank, prev, next_, neighbor_strides, start, image_stride)
    del prev, next_

    filled = value_map[value_rank[:image_stride]]
    filled.shape = (rows + 2, padded_cols)
    return filled[1:-1, 1:-1]
