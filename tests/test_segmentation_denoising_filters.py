from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.segmentation import apply_denoising
from app.models.config import SegParams


def _noisy_uint8(shape=(32, 32), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=128, scale=20, size=shape)
    return np.clip(data, 0, 255).astype(np.uint8)


def _noisy_float(shape=(32, 32), seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


@pytest.mark.parametrize(
    "params, image",
    [
        (SegParams(tv_weight=0.15, tv_eps=1e-4, tv_max_iter=5), _noisy_float()),
        (
            SegParams(
                anisotropic_lambda=0.1,
                anisotropic_kappa=20.0,
                anisotropic_niter=4,
            ),
            _noisy_float(seed=2),
        ),
        (
            SegParams(
                wavelet_sigma=0.05,
                wavelet_mode="soft",
                wavelet_method="BayesShrink",
                wavelet_rescale=False,
            ),
            _noisy_float(seed=3),
        ),
    ],
)
def test_apply_denoising_new_filters_preserve_shape_dtype(params: SegParams, image: np.ndarray) -> None:
    if params.wavelet_sigma > 0:
        pytest.importorskip("pywt")
    result = apply_denoising(image, params)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_apply_denoising_bm3d_runs_when_available() -> None:
    pytest.importorskip("bm3d")
    image = _noisy_float(seed=4)
    params = SegParams(bm3d_enabled=True, bm3d_sigma=0.08, bm3d_stage="hard")
    result = apply_denoising(image, params)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
