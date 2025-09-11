from __future__ import annotations
from typing import Literal
import numpy as np
import cv2


def compute_difference(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    method: Literal["abs", "lab", "edges"] = "abs",
) -> np.ndarray:
    """Compute an enhanced difference image.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (grayscale or BGR, uint8).
    mov : np.ndarray
        Registered moving image (grayscale or BGR, uint8).
    method : {"abs", "lab", "edges"}
        Difference strategy:
        ``"abs"`` uses absolute pixel differences;
        ``"lab"`` diffs the L channel in CIE LAB space with contrast enhancement;
        ``"edges"`` diffs Canny edge maps.

    Returns
    -------
    np.ndarray
        8-bit difference image with contrast enhancement.
    """

    if method == "lab":
        # Convert to LAB and use only the lightness channel
        if ref.ndim == 2 or ref.shape[2] == 1:
            ref_lab = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
            mov_lab = cv2.cvtColor(mov, cv2.COLOR_GRAY2BGR)
        else:
            ref_lab, mov_lab = ref, mov
        ref_lab = cv2.cvtColor(ref_lab, cv2.COLOR_BGR2LAB)
        mov_lab = cv2.cvtColor(mov_lab, cv2.COLOR_BGR2LAB)
        diff = cv2.absdiff(ref_lab[:, :, 0], mov_lab[:, :, 0])
    elif method == "edges":
        # Difference of edge maps
        if ref.ndim == 3 and ref.shape[2] > 1:
            ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            mov_gray = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray, mov_gray = ref, mov
        edges_ref = cv2.Canny(ref_gray, 100, 200)
        edges_mov = cv2.Canny(mov_gray, 100, 200)
        diff = cv2.absdiff(edges_ref, edges_mov)
    else:
        # Absolute difference in intensity
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
