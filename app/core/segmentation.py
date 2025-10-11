from __future__ import annotations
from typing import Mapping, Any
import numpy as np
import cv2
from skimage import morphology, filters


def _seg_param(params: Any, key: str, default: Any) -> Any:
    if hasattr(params, key):
        return getattr(params, key)
    if isinstance(params, Mapping):
        return params.get(key, default)
    return default


def apply_denoising(gray: np.ndarray, params: Any) -> np.ndarray:
    """Apply optional denoising steps configured in :class:`SegParams`.

    Parameters
    ----------
    gray:
        Input grayscale image. The array is never modified in place; a copy is
        returned even when no filters are enabled.
    params:
        Either a :class:`SegParams` instance or a mapping that exposes the
        denoising keys (``gaussian_sigma``, ``median_kernel_size``,
        ``bilateral_diameter``, ``bilateral_sigma_color``,
        ``bilateral_sigma_space``, ``nlm_strength``).
    """

    sigma = float(_seg_param(params, "gaussian_sigma", 0.0) or 0.0)
    median = int(_seg_param(params, "median_kernel_size", 0) or 0)
    bilateral_d = int(_seg_param(params, "bilateral_diameter", 0) or 0)
    bilateral_sigma_color = float(
        _seg_param(params, "bilateral_sigma_color", 0.0) or 0.0
    )
    bilateral_sigma_space = float(
        _seg_param(params, "bilateral_sigma_space", 0.0) or 0.0
    )
    nlm_strength = _seg_param(params, "nlm_strength", None)
    try:
        nlm_strength = None if nlm_strength is None else float(nlm_strength)
    except (TypeError, ValueError):
        nlm_strength = None

    work: np.ndarray | None = None

    def ensure_work() -> np.ndarray:
        nonlocal work
        if work is None:
            work = gray.astype(np.uint8, copy=True)
        return work

    if sigma > 0:
        work = cv2.GaussianBlur(ensure_work(), (0, 0), sigma)

    if median >= 3:
        ksize = median | 1  # ensure odd kernel size
        work = cv2.medianBlur(ensure_work(), ksize)

    if bilateral_d > 0 and (bilateral_sigma_color > 0 or bilateral_sigma_space > 0):
        work = cv2.bilateralFilter(
            ensure_work(),
            bilateral_d,
            max(bilateral_sigma_color, 0.0),
            max(bilateral_sigma_space, 0.0),
        )

    if nlm_strength is not None and nlm_strength > 0:
        work = cv2.fastNlMeansDenoising(
            ensure_work(),
            None,
            float(nlm_strength),
            templateWindowSize=7,
            searchWindowSize=21,
        )

    if work is None:
        return gray.copy()
    if gray.dtype == np.uint8:
        return work
    return work.astype(gray.dtype)


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
    use_clahe: bool = False,
) -> np.ndarray:
    """Segment a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.
    method : str
        Thresholding method: "otsu", "multi_otsu", "li", "yen", "adaptive",
        "local", or "manual".
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
    use_clahe : bool
        Apply CLAHE before thresholding to boost local contrast.
    Examples
    --------
    Segment a low-contrast image using Li's threshold::

        >>> gray = np.full((5, 5), 120, dtype=np.uint8)
        >>> gray[2:, 2:] = 130  # low-contrast foreground
        >>> mask = segment(gray, method="li", invert=False, skip_outline=True)

    Apply Yen's threshold on the same image; this method handles
    low-contrast backgrounds well::

        >>> mask = segment(gray, method="yen", invert=False, skip_outline=True)

    Use Multi-Otsu on an image with three intensity phases; it excels on
    multi-phase images with distinct peaks in their histograms::

        >>> gray = np.concatenate(
        ...     (
        ...         np.full((5, 5), 50, dtype=np.uint8),
        ...         np.full((5, 5), 128, dtype=np.uint8),
        ...         np.full((5, 5), 200, dtype=np.uint8),
        ...     ),
        ...     axis=1,
        ... )
        >>> mask = segment(gray, method="multi_otsu", invert=False, skip_outline=True)

    For full multi-phase segmentation into more than two regions, use
    scikit-image's Multi-Otsu directly::

        >>> from skimage import filters
        >>> img = np.array(
        ...     [[0, 0, 0],
        ...      [128, 128, 128],
        ...      [255, 255, 255]], dtype=np.uint8)
        >>> thresholds = filters.threshold_multiotsu(img, classes=3)
        >>> regions = np.digitize(img, bins=thresholds)
    """

    used_outline = False
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    if method == "manual":
        proc = 255 - gray if invert else gray
        if clahe is not None:
            proc = clahe.apply(proc)
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
        proc = feat
        if clahe is not None:
            proc = clahe.apply(proc)
        if method == "otsu":
            _, th = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw = (th > 0).astype(np.uint8)
        elif method == "adaptive":
            blk = max(3, adaptive_block | 1)
            rng = int(proc.max() - proc.min())
            if rng < 2:
                if use_diff:
                    bw = np.zeros_like(proc, dtype=np.uint8)
                    proc = None
                else:
                    proc = plain
                    if clahe is not None:
                        proc = clahe.apply(proc)
                    rng = int(proc.max() - proc.min())
                    if rng < 2:
                        bw = np.zeros_like(proc, dtype=np.uint8)
                        proc = None
            if proc is not None:
                if use_diff:
                    proc = cv2.normalize(proc, None, 0, 255, cv2.NORM_MINMAX)
                th = cv2.adaptiveThreshold(
                    proc.astype(np.uint8),
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blk,
                    adaptive_C,
                )
                bw = (th > 0).astype(np.uint8)
        elif method == "local":
            blk = max(3, local_block | 1)
            rng = int(proc.max() - proc.min())
            if rng < 2:
                if use_diff:
                    bw = np.zeros_like(proc, dtype=np.uint8)
                    proc = None
                else:
                    proc = plain
                    if clahe is not None:
                        proc = clahe.apply(proc)
                    rng = int(proc.max() - proc.min())
                    if rng < 2:
                        bw = np.zeros_like(proc, dtype=np.uint8)
                        proc = None
            if proc is not None:
                if use_diff:
                    norm = cv2.normalize(proc, None, 0, 255, cv2.NORM_MINMAX)
                    loc = filters.threshold_local(norm, blk)
                    bw = (norm > loc).astype(np.uint8)
                else:
                    loc = filters.threshold_local(proc, blk)
                    bw = (proc > loc).astype(np.uint8)
        elif method == "multi_otsu":
            # ``threshold_multiotsu`` raises a ValueError when the histogram
            # cannot be divided into the requested number of classes (e.g., a
            # uniform image). Other branches of ``segment`` handle such low
            # dynamic range images by returning an empty mask, so mirror that
            # behaviour here instead of letting the exception propagate.
            if np.unique(feat).size < 2:
                bw = np.zeros_like(feat, dtype=np.uint8)
            else:
                try:
                    t = filters.threshold_multiotsu(feat, classes=2)
                except ValueError:
                    bw = np.zeros_like(feat, dtype=np.uint8)
                else:
                    bw = (feat >= t[0]).astype(np.uint8)
        elif method == "li":
            t = filters.threshold_li(feat)
            bw = (proc >= t).astype(np.uint8)
        elif method == "yen":
            t = filters.threshold_yen(feat)
            bw = (proc >= t).astype(np.uint8)
        else:
            t = int(np.clip(manual_thresh, 0, 255))
            bw = (proc >= t).astype(np.uint8)

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
