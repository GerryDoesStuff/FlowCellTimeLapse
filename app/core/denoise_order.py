from __future__ import annotations

from typing import Iterable, List

DEFAULT_DENOISE_ORDER: List[str] = [
    "gaussian",
    "median",
    "bilateral",
    "nlm",
    "tv",
    "anisotropic",
    "wavelet",
    "bm3d",
]


def normalize_denoise_order(order: Iterable[str] | None) -> list[str]:
    """Normalize a denoising step order to include all known steps once."""

    normalized: list[str] = []
    if order is not None:
        for step in order:
            if step in DEFAULT_DENOISE_ORDER and step not in normalized:
                normalized.append(step)
    for step in DEFAULT_DENOISE_ORDER:
        if step not in normalized:
            normalized.append(step)
    return normalized
