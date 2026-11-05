from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fmask import utils
from fmask.satellite import Data
from fmask.unetlib import UNet, fill_batch

BANDS = ["blue", "green", "red", "nir", "swir1"]
PREDICTORS = ["nir", "blue", "swir1"]
CLASSES = ["noncloud", "cloud"]
SHAPE = (70, 83)


def _stub(batch_size=3):
    torch.manual_seed(0)
    cube = (np.random.default_rng(0).random((len(BANDS), *SHAPE)) * 3).astype(np.float32)
    obsmask = np.ones(SHAPE, dtype=bool)
    obsmask[:6] = False
    obsmask[:, -5:] = False
    return SimpleNamespace(
        image=SimpleNamespace(data=Data(cube, BANDS), obsmask=obsmask),
        predictors=PREDICTORS,
        classes=CLASSES,
        patch_size=32,
        patch_stride_classify=24,
        batch_size=batch_size,
        device="cpu",
        # A 1x1 kernel scores each pixel alone, so the patch layout cannot affect the answer.
        model=torch.nn.Conv2d(len(PREDICTORS), len(CLASSES), 1),
    )


def _whole_image_probs(stub, normalize):
    data = stub.image.data
    indices = [data.bands.index(predictor) for predictor in PREDICTORS]
    if normalize:
        cube = utils.normalize_bands(data.data, indices, obsmask=stub.image.obsmask)
    else:
        cube = data.data[indices]
    with torch.no_grad():
        return torch.softmax(stub.model(torch.from_numpy(cube)[None]), dim=1)[0].numpy()


@pytest.mark.parametrize("batch_size", [1, 3, 500], ids=["single", "short_last_batch", "one_batch"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("probability", ["none", "default", "cloud"])
def test_classify_matches_per_pixel_scores(probability, normalize, batch_size):
    stub = _stub(batch_size)
    cube_before = stub.image.data.data.copy()
    mask = stub.image.obsmask
    probs = _whole_image_probs(stub, normalize)

    image_class, image_prob = UNet.classify(stub, probability=probability, normalize=normalize)

    assert image_class.dtype == np.uint8
    assert np.array_equal(image_class[mask], probs.argmax(axis=0)[mask])
    if probability == "none":
        assert image_prob is None
    else:
        assert image_prob.dtype == np.float32
        expected = probs.max(axis=0) if probability == "default" else probs[1]
        assert np.allclose(image_prob[mask], expected[mask], rtol=1e-5, atol=0)
    assert stub.image.data.data.tobytes() == cube_before.tobytes()


def _datacube():
    return torch.from_numpy(
        np.random.default_rng(1).random((3, 70, 83)).astype(np.float32)
    ).to("cpu", dtype=torch.float)


def _stack(datacube, anchors, size):
    return torch.stack([datacube[:, r:r + size, c:c + size] for r, c, _, _ in anchors])


def _anchors(count):
    return [(i * 2, i * 3, 0, 0) for i in range(count)]


def test_fill_batch_matches_stack():
    datacube = _datacube()
    anchors = _anchors(4)
    buffer = torch.empty((4, 3, 32, 32), dtype=datacube.dtype)
    actual = fill_batch(buffer, datacube, anchors, 32)
    expected = _stack(datacube, anchors, 32)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.numpy().tobytes() == expected.numpy().tobytes()


def test_short_last_batch():
    datacube = _datacube()
    buffer = torch.empty((4, 3, 32, 32), dtype=datacube.dtype)
    fill_batch(buffer, datacube, _anchors(4), 32)
    anchors = _anchors(2)
    actual = fill_batch(buffer, datacube, anchors, 32)
    expected = _stack(datacube, anchors, 32)
    assert actual.shape == (2, 3, 32, 32)
    assert actual.numpy().tobytes() == expected.numpy().tobytes()


def test_fill_batch_does_not_change_datacube():
    datacube = _datacube()
    before = datacube.clone()
    buffer = torch.empty((4, 3, 32, 32), dtype=datacube.dtype)
    fill_batch(buffer, datacube, _anchors(4), 32)
    assert torch.equal(datacube, before)
