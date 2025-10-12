from __future__ import annotations
from typing import Mapping, Any
import logging
import numpy as np
import cv2
from skimage import morphology, filters
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
from skimage.util import img_as_float32

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency may be unavailable during tests
    from bm3d import bm3d as bm3d_denoise
except Exception:  # pragma: no cover - import failure is handled at runtime
    bm3d_denoise = None


def _float_to_dtype(arr: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype, copy=False)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scaled = np.rint(arr * info.max).astype(dtype)
        return np.clip(scaled, info.min, info.max)
    return arr.astype(dtype)


def _perona_malik(
    image: np.ndarray,
    iterations: int,
    kappa: float,
    gamma: float,
) -> np.ndarray:
    """Perform Perona–Malik anisotropic diffusion on a grayscale image."""

    img = image.astype(np.float32, copy=True)
    g = float(np.clip(gamma, 0.0, 0.25))
    if g == 0.0 or iterations <= 0 or kappa <= 0.0:
        return img

    for _ in range(iterations):
        north = np.zeros_like(img)
        south = np.zeros_like(img)
        east = np.zeros_like(img)
        west = np.zeros_like(img)

        north[1:, :] = img[1:, :] - img[:-1, :]
        south[:-1, :] = img[:-1, :] - img[1:, :]
        east[:, :-1] = img[:, 1:] - img[:, :-1]
        west[:, 1:] = img[:, :-1] - img[:, 1:]

        c_n = np.exp(-(north / kappa) ** 2)
        c_s = np.exp(-(south / kappa) ** 2)
        c_e = np.exp(-(east / kappa) ** 2)
        c_w = np.exp(-(west / kappa) ** 2)

        img += g * (c_n * north + c_s * south + c_e * east + c_w * west)

    return img


def _seg_param(params: Any, key: str, default: Any) -> Any:
    if hasattr(params, key):
        return getattr(params, key)
    if isinstance(params, Mapping):
        return params.get(key, default)
    return default


def apply_denoising(gray: np.ndarray, params: Any) -> np.ndarray:
    """Apply optional denoising steps configured in :class:`SegParams`.

    The following filters are supported (applied in the listed order when
    enabled): Gaussian blur, median blur, bilateral filtering, fast
    non-local-means, total-variation denoising, Perona–Malik anisotropic
    diffusion, wavelet denoising, and BM3D denoising.
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

    tv_weight = float(_seg_param(params, "tv_weight", 0.0) or 0.0)
    tv_eps = float(_seg_param(params, "tv_eps", 2e-4) or 2e-4)
    tv_max_iter = int(_seg_param(params, "tv_max_iter", 200) or 0)

    anis_lambda = float(_seg_param(params, "anisotropic_lambda", 0.0) or 0.0)
    anis_kappa = float(_seg_param(params, "anisotropic_kappa", 0.0) or 0.0)
    anis_niter = int(_seg_param(params, "anisotropic_niter", 0) or 0)

    wavelet_sigma = _seg_param(params, "wavelet_sigma", 0.0)
    try:
        wavelet_sigma = float(wavelet_sigma)
    except (TypeError, ValueError):
        wavelet_sigma = 0.0
    wavelet_mode = str(_seg_param(params, "wavelet_mode", "soft") or "soft")
    wavelet_rescale = bool(_seg_param(params, "wavelet_rescale", False))
    wavelet_method = str(_seg_param(params, "wavelet_method", "BayesShrink") or "BayesShrink")

    bm3d_enabled = bool(_seg_param(params, "bm3d_enabled", False))
    bm3d_sigma = float(_seg_param(params, "bm3d_sigma", 0.0) or 0.0)
    bm3d_stage = str(_seg_param(params, "bm3d_stage", "hard") or "hard")

    work: np.ndarray | None = None

    def ensure_work() -> np.ndarray:
        nonlocal work
        if work is None:
            work = gray.astype(np.uint8, copy=True)
        return work

    def apply_float_filter(func) -> None:
        nonlocal work
        source = work if work is not None else gray
        float_src = img_as_float32(source)
        result = func(float_src)
        work = _float_to_dtype(result, source.dtype)

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

    if tv_weight > 0 and tv_max_iter > 0:
        apply_float_filter(
            lambda arr: denoise_tv_chambolle(
                arr,
                weight=tv_weight,
                eps=tv_eps,
                max_num_iter=tv_max_iter,
                channel_axis=None,
            )
        )

    if anis_niter > 0 and anis_lambda > 0 and anis_kappa > 0:
        gamma = float(np.clip(anis_lambda, 0.0, 0.25))
        apply_float_filter(lambda arr: _perona_malik(arr, anis_niter, anis_kappa, gamma))

    if wavelet_sigma > 0:
        def _wavelet(arr: np.ndarray) -> np.ndarray:
            sigma_value = wavelet_sigma if wavelet_sigma > 0 else None
            try:
                return denoise_wavelet(
                    arr,
                    sigma=sigma_value,
                    mode=wavelet_mode,
                    rescale_sigma=wavelet_rescale,
                    method=wavelet_method,
                    channel_axis=None,
                ).astype(np.float32, copy=False)
            except ImportError:
                logger.warning(
                    "Wavelet denoising requested but PyWavelets is not installed."
                )
                return arr

        apply_float_filter(_wavelet)

    if bm3d_enabled and bm3d_sigma > 0:
        if bm3d_denoise is None:
            logger.warning(
                "BM3D denoising requested but the optional 'bm3d' package is not installed."
            )
        else:
            sigma_value = bm3d_sigma
            if sigma_value > 1.0:
                sigma_value = sigma_value / 255.0
            sigma_value = float(np.clip(sigma_value, 0.0, 1.0))

            def _bm3d(arr: np.ndarray) -> np.ndarray:
                try:
                    return bm3d_denoise(arr, sigma_psd=sigma_value, stage=bm3d_stage)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception("BM3D denoising failed: %s", exc)
                    return arr

            apply_float_filter(_bm3d)

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
