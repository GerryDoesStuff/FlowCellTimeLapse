from __future__ import annotations
import numpy as np
import cv2
from typing import List

def estimate_temporal_background(frames: list[np.ndarray], n_early: int=5) -> np.ndarray:
    # Use median of the earliest n frames as background proxy
    n = min(n_early, len(frames))
    stack = np.stack(frames[:n], axis=0).astype(np.float32)
    med = np.median(stack, axis=0).astype(np.uint8)
    return med

def normalize_background(img: np.ndarray, bg: np.ndarray, rescale: bool = True) -> np.ndarray:
    """Subtract a background image and optionally rescale the result.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    bg : np.ndarray
        Background image to subtract.
    rescale : bool, optional
        If ``True`` (default), the difference image is rescaled to the full
        ``[0, 255]`` range using :func:`cv2.normalize`. When ``False``, the
        raw difference is used and negative values are clipped to ``0``.

    Returns
    -------
    np.ndarray
        Background corrected image as ``uint8``.
    """

    # Compute difference in a signed type so negatives can be detected.
    diff = img.astype(np.float32) - bg.astype(np.float32)

    if rescale:
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # When not rescaling, simply clamp negative values to zero.
        diff = np.clip(diff, 0, None)

    return diff.astype(np.uint8)
