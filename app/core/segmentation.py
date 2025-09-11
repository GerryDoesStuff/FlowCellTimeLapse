from __future__ import annotations
import numpy as np
import cv2
from skimage import morphology, filters


def outline_focused(gray: np.ndarray, invert: bool = True) -> np.ndarray:
    """Enhance outlines via a black-hat transform."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    if invert:
        g = 255 - gray
    else:
        g = gray
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    return blackhat


def segment(
    gray: np.ndarray,
    method: str = "otsu",
    invert: bool = True,
    skip_outline: bool = False,
    use_diff: bool = False,
    auto_skip_outline: bool = True,
    manual_thresh: int = 128,
    adaptive_block: int = 51,
    adaptive_C: int = 5,
    local_block: int = 51,
    morph_open_radius: int | None = None,
    morph_close_radius: int | None = None,
    remove_objects_smaller_px: int = 0,
    remove_holes_smaller_px: int = 0,
) -> np.ndarray:
    """Segment a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.
    method : str
        Thresholding method: "otsu", "adaptive", "local", or "manual".
    invert : bool
        Treat cells as darker than background.
    skip_outline : bool
        If True, bypass the black-hat outline prefilter. Useful for
        low-contrast images where the prefilter may remove small features.
    use_diff : bool
        Input image is a frame-to-frame difference; black-hat is skipped
        because edges are already enhanced.
    auto_skip_outline : bool
        If True, automatically bypass the black-hat step when it produces
        almost no signal (heuristic based on mean intensity).
    manual_thresh, adaptive_block, adaptive_C, local_block, morph_open_radius,
    morph_close_radius, remove_objects_smaller_px, remove_holes_smaller_px :
        Parameters controlling thresholding and post-processing.
    """

    used_outline = False

    if method == "manual":
        proc = 255 - gray if invert else gray
        t = int(np.clip(manual_thresh, 0, 255))
        bw = (proc >= t).astype(np.uint8)
    else:
        plain = 255 - gray if invert else gray
        if skip_outline or use_diff:
            feat = plain
        else:
            bh = outline_focused(gray, invert=invert)
            # If the outline-focused image lacks dynamic range,
            # it likely washed out features. Fall back to the plain image.
            if auto_skip_outline and (bh.std() < 1 or bh.max() < 2):
                feat = plain
            else:
                feat = bh
                used_outline = True
        if method == "otsu":
            _, th = cv2.threshold(feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw = (th > 0).astype(np.uint8)
        elif method == "adaptive":
            blk = max(3, adaptive_block | 1)
            rng = int(feat.max() - feat.min())
            if rng < 2:
                if use_diff:
                    bw = np.zeros_like(feat, dtype=np.uint8)
                    feat = None
                else:
                    feat = plain
                    rng = int(feat.max() - feat.min())
                    if rng < 2:
                        bw = np.zeros_like(feat, dtype=np.uint8)
                        feat = None
            if feat is not None:
                if use_diff:
                    feat = cv2.normalize(feat, None, 0, 255, cv2.NORM_MINMAX)
                th = cv2.adaptiveThreshold(
                    feat.astype(np.uint8),
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blk,
                    adaptive_C,
                )
                bw = (th > 0).astype(np.uint8)
        elif method == "local":
            blk = max(3, local_block|1)
            rng = int(feat.max() - feat.min())
            if rng < 2:
                if use_diff:
                    bw = np.zeros_like(feat, dtype=np.uint8)
                    feat = None
                else:
                    feat = plain
                    rng = int(feat.max() - feat.min())
                    if rng < 2:
                        bw = np.zeros_like(feat, dtype=np.uint8)
                        feat = None
            if feat is not None:
                if use_diff:
                    feat = cv2.normalize(feat, None, 0, 255, cv2.NORM_MINMAX)
                loc = filters.threshold_local(feat, blk)
                bw = (feat > loc).astype(np.uint8)
        else:
            t = int(np.clip(manual_thresh, 0, 255))
            bw = (feat >= t).astype(np.uint8)

    # Morphology: closing before opening. When outline-based thresholds are used
    # (the default path), radii default to zero, avoiding unnecessary smoothing.
    # For non-outline paths, restore small default radii for basic cleanup.
    if morph_close_radius is None:
        morph_close_radius = 0 if used_outline else 2
    if morph_open_radius is None:
        morph_open_radius = 0 if used_outline else 2

    if morph_close_radius > 0:
        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_radius * 2 + 1, morph_close_radius * 2 + 1)
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se)
    if morph_open_radius > 0:
        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_open_radius * 2 + 1, morph_open_radius * 2 + 1)
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, se)

    if remove_objects_smaller_px>0:
        bw = morphology.remove_small_objects(bw.astype(bool), remove_objects_smaller_px).astype(np.uint8)
    if remove_holes_smaller_px>0:
        bw = morphology.remove_small_holes(bw.astype(bool), remove_holes_smaller_px).astype(np.uint8)

    return (bw>0).astype(np.uint8)
