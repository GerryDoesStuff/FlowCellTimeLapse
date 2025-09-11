from __future__ import annotations
import numpy as np
import cv2


def compute_difference(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Compute an enhanced absolute difference image.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (grayscale, uint8).
    mov : np.ndarray
        Registered moving image (grayscale, uint8).

    Returns
    -------
    np.ndarray
        8-bit difference image with contrast enhancement.
    """
    # Absolute difference
    diff = cv2.absdiff(ref, mov)

    # Normalize to full 0-255 range
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff = diff.astype(np.uint8)

    # Optional CLAHE for local contrast enhancement
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        diff = clahe.apply(diff)
    except Exception:
        # If CLAHE creation fails, fall back to normalized difference
        pass

    return diff
