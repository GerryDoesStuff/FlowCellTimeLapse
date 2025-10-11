from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Optional, Dict
from .io_utils import imread_gray, imread_color, ensure_dir
from .registration import register_ecc, register_orb, register_orb_ecc, crop_to_overlap, preprocess
from .segmentation import segment, apply_denoising
from .difference import compute_difference
from .background import normalize_background, estimate_temporal_background
from .evaluation import write_shape_properties
import logging
import re

logger = logging.getLogger(__name__)


def overlay_outline(
    gray: np.ndarray,
    mask: np.ndarray | None = None,
    new_mask: np.ndarray | None = None,
    lost_mask: np.ndarray | None = None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    new_color: tuple[int, int, int] = (0, 255, 0),
    lost_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Overlay segmentation outlines on a grayscale image.

    Parameters
    ----------
    gray:
        Base grayscale image.
    mask:
        Optional mask to outline in ``color``.
    new_mask:
        Regions present in the current frame but absent in the previous.
    lost_mask:
        Regions present in the previous frame but absent in the current.
    color, new_color, lost_color:
        BGR colors used for the respective masks.
    """

    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _draw(m: np.ndarray | None, col: tuple[int, int, int]) -> None:
        if m is None:
            return
        contours, _ = cv2.findContours(
            (m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(result, contours, -1, col, 1)

    _draw(mask, color)
    _draw(new_mask, new_color)
    _draw(lost_mask, lost_color)
    return result


def _detect_green_magenta(
    gm_composite: np.ndarray,
    prev_seg: np.ndarray,
    curr_seg: np.ndarray,
    app_cfg: dict,
    *,
    direction: str,
    diagnostics_dir: Path | None = None,
    frame_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect green and magenta difference masks.

    The ``gm_composite`` image encodes a weighted blend of the previous frame
    in the green channel and the current frame in red/blue (magenta).  The
    blending weight ``alpha`` is derived from
    ``app_cfg['gm_opacity']`` (percentage of the current frame, default
    ``50``) such that the previous frame contributes ``1 - alpha``. Depending on
    ``direction`` the roles of these colors are swapped so that ``green`` always
    represents the frame that is considered "lost" and ``magenta`` the frame
    considered "new".

    Parameters
    ----------
    gm_composite:
        BGR image containing the green–magenta composite.
    prev_seg, curr_seg:
        Segmentation masks of the previous and current frame respectively.
    app_cfg:
        Application configuration dictionary providing thresholding and
        morphology parameters. When ``gm_thresh_method`` is set to
        ``"percentile"``, the percentiles used for magenta and green channels
        can be specified independently via ``gm_thresh_percentile_magenta`` and
        ``gm_thresh_percentile_green`` (falling back to
        ``gm_thresh_percentile`` if not provided).
    direction:
        Either ``"first-to-last"`` or ``"last-to-first"``. Included for API
        compatibility; callers running in reverse must handle any swapping of
        the returned masks themselves.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``green_mask`` and ``magenta_mask`` after segmentation filtering.
    """

    # ``direction`` is currently unused; swapping is handled by callers.
    _ = direction

    lab = cv2.cvtColor(gm_composite, cv2.COLOR_BGR2LAB)
    a_channel = lab[..., 1].astype(np.int16) - 128  # center at 0
    sat = float(app_cfg.get("gm_saturation", 1.0))
    a_channel = np.clip(a_channel * sat, -255, 255).astype(np.int16)
    abs_a = np.abs(a_channel).astype(np.uint8)
    if diagnostics_dir is not None and frame_index is not None:
        ensure_dir(diagnostics_dir)
        cv2.imencode(".png", abs_a)[1].tofile(
            str(diagnostics_dir / f"{frame_index:04d}_a.png")
        )

    magenta_vals = a_channel[a_channel > 0].astype(np.uint8)
    green_vals = (-a_channel[a_channel < 0]).astype(np.uint8)
    thresh_method = str(app_cfg.get("gm_thresh_method", "otsu")).lower()
    if thresh_method == "percentile":
        perc_magenta = float(
            app_cfg.get(
                "gm_thresh_percentile_magenta",
                app_cfg.get("gm_thresh_percentile", 99.0),
            )
        )
        perc_green = float(
            app_cfg.get(
                "gm_thresh_percentile_green",
                app_cfg.get("gm_thresh_percentile", 99.0),
            )
        )
        gm_thresh_magenta = (
            int(np.percentile(magenta_vals, perc_magenta))
            if magenta_vals.size > 0
            else 0
        )
        gm_thresh_green = (
            int(np.percentile(green_vals, perc_green))
            if green_vals.size > 0
            else 0
        )
    else:
        gm_thresh_magenta = gm_thresh_green = int(
            cv2.threshold(abs_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        )

    gm_thresh_magenta = max(gm_thresh_magenta, 1)
    gm_thresh_green = max(gm_thresh_green, 1)

    if diagnostics_dir is not None and frame_index is not None:
        if magenta_vals.size > 0:
            hist_m, _ = np.histogram(magenta_vals, bins=256, range=(0, 256))
            above_m = (
                int(hist_m[gm_thresh_magenta + 1 :].sum())
                if gm_thresh_magenta < 255
                else 0
            )
            max_magenta = int(magenta_vals.max())
        else:
            above_m = 0
            max_magenta = 0
        if green_vals.size > 0:
            hist_g, _ = np.histogram(green_vals, bins=256, range=(0, 256))
            above_g = (
                int(hist_g[gm_thresh_green + 1 :].sum())
                if gm_thresh_green < 255
                else 0
            )
            max_green = int(green_vals.max())
        else:
            above_g = 0
            max_green = 0
        logger.debug(
            (
                "Frame %04d: gm_thresh_magenta=%d gm_thresh_green=%d "
                "max_magenta=%d max_green=%d pixels_above_magenta_thresh=%d "
                "pixels_above_green_thresh=%d"
            ),
            frame_index,
            gm_thresh_magenta,
            gm_thresh_green,
            max_magenta,
            max_green,
            above_m,
            above_g,
        )

    green_mask = (a_channel < -gm_thresh_green).astype(np.uint8)
    magenta_mask = (a_channel > gm_thresh_magenta).astype(np.uint8)

    close_k = int(app_cfg.get("gm_close_kernel", 0))
    if close_k > 0:
        kernel = np.ones((close_k, close_k), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_CLOSE, kernel)

    dilate_k = int(app_cfg.get("gm_dilate_kernel", 0))
    if dilate_k > 0:
        kernel = np.ones((dilate_k, dilate_k), np.uint8)
        green_mask = cv2.dilate(green_mask, kernel)
        magenta_mask = cv2.dilate(magenta_mask, kernel)

    min_seg_overlap = float(app_cfg.get("component_min_seg_overlap", 0.5))

    def _mask_with_seg(diff: np.ndarray, seg: np.ndarray) -> np.ndarray:
        filtered = np.zeros_like(diff)
        num, labels = cv2.connectedComponents(diff)
        for lbl in range(1, num):
            comp = (labels == lbl).astype(np.uint8)
            area = int(comp.sum())
            if area == 0:
                continue
            overlap = int((comp & seg).sum())
            if overlap / area >= min_seg_overlap:
                filtered |= comp & seg
        return filtered

    green_mask = _mask_with_seg(green_mask, prev_seg)
    magenta_mask = _mask_with_seg(magenta_mask, curr_seg)

    return green_mask, magenta_mask


def analyze_sequence(paths: List[Path], reg_cfg: dict, seg_cfg: dict, app_cfg: dict, out_dir: Path) -> pd.DataFrame:
    norm = bool(app_cfg.get("normalize", True))
    scale_minmax = app_cfg.get("scale_minmax")
    imgs_gray = [imread_gray(p, normalize=norm, scale_minmax=scale_minmax) for p in paths]
    H, W = imgs_gray[0].shape[:2]

    logger.info("Loaded %d images with dimensions %dx%d", len(imgs_gray), H, W)

    direction = app_cfg.get("direction", "last-to-first")
    ordered_indices = list(range(len(paths)))
    if direction == "last-to-first":
        ordered_indices.reverse()

        def _extract_num(p: Path) -> Optional[int]:
            m = re.search(r"\d+", p.name)
            return int(m.group()) if m else None

        start_num = _extract_num(paths[ordered_indices[0]])
        last_num = _extract_num(paths[-1])
        if start_num is not None and last_num is not None and start_num < last_num:
            logger.warning(
                "Input paths may be unsorted: starting frame %s is lower-numbered than last path %s",
                paths[ordered_indices[0]].name,
                paths[-1].name,
            )
    ref_idx = ordered_indices[0]
    logger.info(
        "Starting analysis with direction=%s reference_frame_index=%d",
        direction,
        ref_idx,
    )

    subtract_bg = bool(app_cfg.get("subtract_background", False))
    logger.info("Background subtraction %s", "enabled" if subtract_bg else "disabled")

    if subtract_bg:
        bg = estimate_temporal_background(imgs_gray, n_early=5)
        imgs_norm = [normalize_background(g, bg) for g in imgs_gray]
    else:
        imgs_norm = imgs_gray
    gauss_sigma = float(reg_cfg.get("gauss_blur_sigma", 1.0))
    clahe_clip = float(reg_cfg.get("clahe_clip", 2.0))
    clahe_grid = int(reg_cfg.get("clahe_grid", 8))
    use_clahe = bool(reg_cfg.get("use_clahe", True))
    initial_radius = int(reg_cfg.get("initial_radius", min(H, W) // 2))
    growth_factor = float(reg_cfg.get("growth_factor", 1.0))
    logger.info(
        "Preprocessing parameters: gauss_sigma=%.2f use_clahe=%s clahe_clip=%.2f clahe_grid=%d initial_radius=%d growth_factor=%.2f",
        gauss_sigma,
        use_clahe,
        clahe_clip,
        clahe_grid,
        initial_radius,
        growth_factor,
    )
    imgs_norm = [preprocess(g, gauss_sigma, clahe_clip, clahe_grid, use_clahe) for g in imgs_norm]

    save_diagnostics = bool(app_cfg.get("save_diagnostics", True))

    ensure_dir(out_dir)
    reg_dir = out_dir / "registered"; ensure_dir(reg_dir)
    mov_dir = reg_dir / "mov"; ensure_dir(mov_dir)
    prev_dir = reg_dir / "prev"
    if save_diagnostics:
        ensure_dir(prev_dir)

    diff_dir = out_dir / "diff"; ensure_dir(diff_dir)
    diff_raw_dir = diff_dir / "raw"; ensure_dir(diff_raw_dir)
    diff_bw_dir = diff_dir / "bw"; ensure_dir(diff_bw_dir)
    diff_gm_dir = diff_dir / "gm"
    diff_green_dir = diff_dir / "green"; ensure_dir(diff_green_dir)
    diff_magenta_dir = diff_dir / "magenta"; ensure_dir(diff_magenta_dir)

    diff_new_dir = diff_dir / "new"
    diff_lost_dir = diff_dir / "lost"
    diff_gain_dir = diff_dir / "gain"
    diff_loss_dir = diff_dir / "loss"
    diff_overlap_dir = diff_dir / "overlap"
    diff_union_dir = diff_dir / "union"
    diff_a_dir = diff_dir / "a_channel"
    if save_diagnostics:
        ensure_dir(diff_new_dir)
        ensure_dir(diff_lost_dir)
        ensure_dir(diff_gain_dir)
        ensure_dir(diff_loss_dir)
        ensure_dir(diff_overlap_dir)
        ensure_dir(diff_union_dir)
        ensure_dir(diff_a_dir)
        ensure_dir(diff_gm_dir)

    overlay_dir = out_dir / "overlay"
    seg_dir = out_dir / "seg"
    if save_diagnostics:
        ensure_dir(overlay_dir)
        ensure_dir(seg_dir)

    gm_opacity = int(app_cfg.get("gm_opacity", 50))

    rows: List[Dict] = []

    step_transforms: Dict[int, np.ndarray] = {ref_idx: np.eye(3, dtype=np.float32)}
    registered_frames: Dict[int, np.ndarray] = {ref_idx: imgs_norm[ref_idx]}
    valid_masks: Dict[int, np.ndarray] = {}

    if initial_radius > 0:
        cx, cy = W // 2, H // 2
        x0 = max(cx - initial_radius, 0)
        y0 = max(cy - initial_radius, 0)
        x1 = min(cx + initial_radius, W)
        y1 = min(cy + initial_radius, H)
        mask_area = (x1 - x0) * (y1 - y0)
        min_area = int(0.25 * H * W)
        if mask_area < min_area:
            msg = (
                f"Initial radius {initial_radius} covers only {mask_area} "
                f"pixels which is below 25% of the frame ({min_area})"
            )
            logger.error(msg)
            raise ValueError(msg)
        global_mask = np.zeros((H, W), dtype=np.uint8)
        global_mask[y0:y1, x0:x1] = 255
    else:
        global_mask = np.ones((H, W), dtype=np.uint8) * 255

    # Phase 1: register frames and track per-frame crop rectangles
    for idx, k in enumerate(ordered_indices):
        g_full = imgs_norm[k]
        logger.debug("Frame %d: registration phase", k)
        if idx == 0:
            valid_mask = global_mask.copy()
            registered_frames[k] = g_full
        else:
            prev_k = ordered_indices[idx - 1]
            ref_gray = imgs_norm[prev_k]
            method = reg_cfg.get("method", "ECC").upper()
            logger.debug("Frame %d: registration method %s", k, method)
            if method == "ORB":
                success, W_step, _, valid_mask, fb, _, _ = register_orb(
                    ref_gray,
                    g_full,
                    model=reg_cfg.get("model", "homography"),
                    orb_features=int(reg_cfg.get("orb_features", 4000)),
                    match_ratio=float(reg_cfg.get("match_ratio", 0.75)),
                    min_keypoints=int(reg_cfg.get("min_keypoints", 8)),
                    min_matches=int(reg_cfg.get("min_matches", 8)),
                    use_ecc_fallback=bool(reg_cfg.get("use_ecc_fallback", True)),
                )
                if fb:
                    logger.warning("ORB registration fell back to ECC at frame %d", k)
            elif method == "ORB+ECC":
                success, W_step, _, valid_mask, _, _ = register_orb_ecc(
                    ref_gray,
                    g_full,
                    model=reg_cfg.get("model", "affine"),
                    max_iters=int(reg_cfg.get("max_iters", 1000)),
                    eps=float(reg_cfg.get("eps", 1e-6)),
                    orb_features=int(reg_cfg.get("orb_features", 4000)),
                    match_ratio=float(reg_cfg.get("match_ratio", 0.75)),
                    min_keypoints=int(reg_cfg.get("min_keypoints", 8)),
                    min_matches=int(reg_cfg.get("min_matches", 8)),
                    use_ecc_fallback=bool(reg_cfg.get("use_ecc_fallback", True)),
                    mask=global_mask if reg_cfg.get("use_masked_ecc", True) else None,
                )
            else:
                success, W_step, _, valid_mask = register_ecc(
                    ref_gray,
                    g_full,
                    model=reg_cfg.get("model", "affine"),
                    max_iters=int(reg_cfg.get("max_iters", 1000)),
                    eps=float(reg_cfg.get("eps", 1e-6)),
                    mask=global_mask if reg_cfg.get("use_masked_ecc", True) else None,
                )
            if not success:
                logger.warning("Registration failed at frame %d", k)
                step_transforms[k] = step_transforms[prev_k].copy()
                registered_frames[k] = cv2.warpPerspective(
                    imgs_norm[k], step_transforms[k], (W, H)
                )
                valid_masks[k] = np.ones((H, W), dtype=np.uint8)
                continue
            W_h = W_step if W_step.shape == (3, 3) else np.vstack([W_step, [0, 0, 1]])
            step_transforms[k] = step_transforms[prev_k] @ W_h
            registered_frames[k] = cv2.warpPerspective(
                imgs_norm[k], step_transforms[k], (W, H)
            )
            valid_masks[k] = valid_mask

    expanded_mask = (global_mask > 0).astype(np.uint8)
    dilate_radius = int(reg_cfg.get("mask_dilate_radius", 0))
    kernel = None
    if dilate_radius > 0:
        ksize = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    for idx, k in enumerate(ordered_indices[1:], start=1):
        vm = valid_masks.get(k)
        if vm is None:
            continue
        prev_k = ordered_indices[idx - 1]
        T_prev = step_transforms.get(prev_k, np.eye(3, dtype=np.float32))
        vm_ref = cv2.warpPerspective(vm, T_prev, (W, H))
        if kernel is not None:
            vm_ref = cv2.dilate(vm_ref, kernel)
        expanded_mask = cv2.bitwise_and(expanded_mask, vm_ref)

    crop_rect = crop_to_overlap(expanded_mask)
    crop_rects: Dict[int, tuple[int, int, int, int]] = {k: crop_rect for k in ordered_indices}
    x_f, y_f, w_f, h_f = crop_rect
    overlap_area = w_f * h_f
    min_overlap = int(0.01 * H * W)
    if overlap_area < min_overlap:
        msg = f"Final overlap area {overlap_area} below 1% threshold {min_overlap}"
        logger.error(msg)
        raise ValueError(msg)
    # Phase 2: segmentation using per-frame masks
    def _save_mask(
        idx: int,
        mask: np.ndarray,
        x_off: int,
        y_off: int,
        *,
        target_dir: Path,
        suffix: str = "",
        overlay: bool = True,
    ) -> None:
        h, w = mask.shape[:2]
        full_mask = np.zeros_like(imgs_gray[idx], dtype=mask.dtype)
        full_mask[y_off : y_off + h, x_off : x_off + w] = mask
        cv2.imencode(".png", (full_mask * 255).astype(np.uint8))[1].tofile(
            str(target_dir / f"mask_{idx:04d}{suffix}.png")
        )
        if overlay:
            frame_color = cv2.cvtColor(imgs_gray[idx], cv2.COLOR_GRAY2BGR)
            cnts, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts:
                x_m, y_m, w_m, h_m = cv2.boundingRect(np.vstack(cnts))
                cv2.rectangle(
                    frame_color,
                    (x_off + x_m, y_off + y_m),
                    (x_off + x_m + w_m, y_off + y_m + h_m),
                    (0, 255, 0),
                    1,
                )
            cv2.imencode(".png", frame_color)[1].tofile(
                str(target_dir / f"mask_{idx:04d}_overlay{suffix}.png")
            )

    ref_gray = registered_frames[ref_idx]
    ref_input = apply_denoising(ref_gray, seg_cfg)
    bw_ref = segment(
        ref_input,
        method=seg_cfg.get("method", "otsu"),
        invert=bool(seg_cfg.get("invert", True)),
        skip_outline=bool(seg_cfg.get("skip_outline", False)),
        manual_thresh=int(seg_cfg.get("manual_thresh", 128)),
        adaptive_block=int(seg_cfg.get("adaptive_block", 51)),
        adaptive_C=int(seg_cfg.get("adaptive_C", 5)),
        local_block=int(seg_cfg.get("local_block", 51)),
        morph_open_radius=(
            int(seg_cfg["morph_open_radius"])
            if seg_cfg.get("morph_open_radius") is not None
            else None
        ),
        morph_close_radius=(
            int(seg_cfg["morph_close_radius"])
            if seg_cfg.get("morph_close_radius") is not None
            else None
        ),
        remove_objects_smaller_px=int(seg_cfg.get("remove_objects_smaller_px", 64)),
        remove_holes_smaller_px=int(seg_cfg.get("remove_holes_smaller_px", 64)),
        use_clahe=bool(seg_cfg.get("use_clahe", False)),
    )

    # store previous frame, full segmentation mask, and index for iterative
    # processing. ``prev_full_seg`` represents the complete segmentation of the
    # previously processed frame so that regions outside the current crop are
    # preserved across iterations.
    prev_gray = ref_gray
    prev_full_seg = bw_ref
    prev_k = ref_idx

    ecc_mask = None
    all_masks_empty = not np.any(bw_ref)

    for idx, k in enumerate(ordered_indices):
        logger.debug("Frame %d: segmentation phase", k)
        x_k, y_k, w_k, h_k = crop_rects.get(k, (0, 0, W, H))
        prev_crop = prev_gray[y_k:y_k + h_k, x_k:x_k + w_k]
        prev_full_seg_crop = prev_full_seg[y_k:y_k + h_k, x_k:x_k + w_k]
        T = step_transforms.get(k, np.eye(3, dtype=np.float32))
        warped = registered_frames.get(k)
        if warped is None:
            warped = cv2.warpPerspective(imgs_norm[k], T, (W, H))
            registered_frames[k] = warped
        mov_crop = warped[y_k:y_k + h_k, x_k:x_k + w_k]

        # Create a green–magenta composite to highlight differences, blending
        # the frames according to ``gm_opacity``. The previous frame contributes
        # ``1 - alpha`` while the current frame contributes ``alpha``.
        alpha = gm_opacity / 100.0
        gm_composite = np.stack(
            [
                (mov_crop * alpha).astype(np.uint8),  # blue
                (prev_crop * (1 - alpha)).astype(np.uint8),  # green
                (mov_crop * alpha).astype(np.uint8),  # red
            ],
            axis=-1,
        )

        if save_diagnostics:
            gm_disp = gm_composite.copy()
            sat = float(app_cfg.get("gm_saturation", 1.0))
            if sat != 1.0:
                lab = cv2.cvtColor(gm_disp, cv2.COLOR_BGR2LAB).astype(np.int16)
                a = lab[..., 1].astype(np.int16) - 128
                a = np.clip(a * sat, -255, 255) + 128
                lab[..., 1] = a.astype(np.uint8)
                gm_disp = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            cv2.imencode(".png", gm_disp)[1].tofile(
                str(diff_gm_dir / f"{k:04d}_gm.png")
            )

        seg_img = None
        bw_diff = None
        if idx > 0:
            seg_img = compute_difference(
                prev_crop, mov_crop, method=app_cfg.get("difference_method", "abs")
            )
            seg_input = apply_denoising(seg_img, seg_cfg)
            bw_diff = segment(
                seg_input,
                method=seg_cfg.get("method", "otsu"),
                invert=bool(seg_cfg.get("invert", True)),
                skip_outline=bool(seg_cfg.get("skip_outline", False)),
                use_diff=True,
                manual_thresh=int(seg_cfg.get("manual_thresh", 128)),
                adaptive_block=int(seg_cfg.get("adaptive_block", 51)),
                adaptive_C=int(seg_cfg.get("adaptive_C", 5)),
                local_block=int(seg_cfg.get("local_block", 51)),
                morph_open_radius=(
                    int(seg_cfg["morph_open_radius"])
                    if seg_cfg.get("morph_open_radius") is not None
                    else None
                ),
                morph_close_radius=(
                    int(seg_cfg["morph_close_radius"])
                    if seg_cfg.get("morph_close_radius") is not None
                    else None
                ),
                remove_objects_smaller_px=int(
                    seg_cfg.get("remove_objects_smaller_px", 64)
                ),
                remove_holes_smaller_px=int(
                    seg_cfg.get("remove_holes_smaller_px", 64)
                ),
                use_clahe=bool(seg_cfg.get("use_clahe", False)),
            )
            cv2.imencode(".png", seg_img)[1].tofile(
                str(diff_raw_dir / f"{k:04d}_diff.png")
            )
            bw_diff_u8 = (bw_diff * 255).astype(np.uint8)
            cv2.imencode(".png", bw_diff_u8)[1].tofile(
                str(diff_bw_dir / f"{k:04d}_bw_diff.png")
            )
            write_shape_properties(bw_diff_u8, diff_bw_dir / "bw_props.csv")

        # Use the difference mask directly for subsequent processing. When
        # ``idx == 0`` no difference is available, so fall back to an empty
        # mask and skip gain/loss calculations.
        if bw_diff is not None:
            seg_mask = bw_diff
        else:
            seg_mask = np.zeros_like(prev_full_seg_crop)

        if save_diagnostics:
            _save_mask(k, seg_mask, x_k, y_k, target_dir=seg_dir)

            if bw_diff is not None:
                _save_mask(
                    k,
                    seg_mask,
                    x_k,
                    y_k,
                    target_dir=diff_bw_dir,
                    suffix="_difference",
                    overlay=False,
                )

        # Obtain masks highlighting regions unique to the previous (green) and
        # current (magenta) frame. Masks are returned without swapping; when
        # processing in reverse we swap after saving so classification remains
        # direction-independent.
        green_mask, magenta_mask = _detect_green_magenta(
            gm_composite,
            prev_full_seg_crop,
            seg_mask,
            app_cfg,
            direction=direction,
            diagnostics_dir=diff_a_dir if save_diagnostics else None,
            frame_index=k,
        )
        if save_diagnostics:
            gm_mask_u8 = ((green_mask | magenta_mask) * 255).astype(np.uint8)
            write_shape_properties(gm_mask_u8, diff_gm_dir / "gm_props.csv")

        # Prepare updated segmentation for the current frame before any
        # direction-dependent swapping occurs so that stable regions persist.
        updated_crop = (prev_full_seg_crop & (~green_mask)) | magenta_mask
        prev_area_px = int(prev_full_seg_crop.sum())
        curr_seg = updated_crop if idx > 0 else prev_full_seg_crop.copy()
        bw_overlap = (prev_full_seg_crop & curr_seg).astype(np.uint8)
        bw_union = (prev_full_seg_crop | curr_seg).astype(np.uint8)

        # Suppress spurious "new" detections when the difference mask matches
        # the previous segmentation (e.g., pure intensity changes).
        if np.array_equal(seg_mask, prev_full_seg_crop):
            magenta_mask = np.zeros_like(magenta_mask)
            if np.any(curr_seg):
                green_mask = np.zeros_like(green_mask)

        if idx > 0:
            green_u8 = (green_mask * 255).astype(np.uint8)
            magenta_u8 = (magenta_mask * 255).astype(np.uint8)
            cv2.imencode('.png', green_u8)[1].tofile(
                str(diff_green_dir / f"{prev_k:04d}_bw_green.png")
            )
            cv2.imencode('.png', magenta_u8)[1].tofile(
                str(diff_magenta_dir / f"{prev_k:04d}_bw_magenta.png")
            )
            write_shape_properties(green_u8, diff_green_dir / "green_props.csv")
            write_shape_properties(magenta_u8, diff_magenta_dir / "magenta_props.csv")
            if save_diagnostics:
                cv2.imencode('.png', (bw_overlap * 255).astype(np.uint8))[1].tofile(
                    str(diff_overlap_dir / f"{prev_k:04d}_bw_overlap.png")
                )
                cv2.imencode('.png', (bw_union * 255).astype(np.uint8))[1].tofile(
                    str(diff_union_dir / f"{prev_k:04d}_bw_union.png")
                )

        if direction == "last-to-first":
            prev_full_seg_crop, curr_seg = curr_seg, prev_full_seg_crop
            green_mask, magenta_mask = magenta_mask, green_mask

        if not np.any(curr_seg):
            logger.warning(
                "Frame %d: segmentation mask is empty; skipping ecc_mask update", k
            )
            ecc_mask = None
        else:
            all_masks_empty = False
            ecc_mask = curr_seg.copy()

        # Dilate segmentation masks slightly before classifying regions as
        # "new" or "lost" to tolerate small registration errors.
        class_dilate_k = int(app_cfg.get("class_dilate_kernel", 0))
        if class_dilate_k > 0:
            kernel = np.ones((class_dilate_k, class_dilate_k), np.uint8)
            prev_dilated = cv2.dilate(prev_full_seg_crop, kernel)
            seg_dilated = cv2.dilate(curr_seg, kernel)
        else:
            prev_dilated = prev_full_seg_crop
            seg_dilated = curr_seg

        # Classify connected components based on their overlap with the
        # opposite frame. A component is only considered gained or lost if the
        # overlap ratio falls below a configurable threshold.
        min_overlap = float(app_cfg.get("component_min_overlap", 0.5))

        bw_lost = np.zeros_like(prev_full_seg_crop)
        num_prev, labels_prev = cv2.connectedComponents(prev_full_seg_crop)
        for lbl in range(1, num_prev):
            comp = (labels_prev == lbl).astype(np.uint8)
            area = int(comp.sum())
            if area == 0:
                continue
            overlap = int((comp & seg_dilated).sum())
            if overlap / area < min_overlap:
                bw_lost |= comp & green_mask

        bw_new = np.zeros_like(curr_seg)
        num_curr, labels_curr = cv2.connectedComponents(curr_seg)
        for lbl in range(1, num_curr):
            comp = (labels_curr == lbl).astype(np.uint8)
            area = int(comp.sum())
            if area == 0:
                continue
            overlap = int((comp & prev_dilated).sum())
            if overlap / area < min_overlap:
                bw_new |= comp & magenta_mask

        if bw_diff is not None:
            gain_mask = (bw_diff & bw_new).astype(np.uint8)
            loss_mask = (bw_diff & bw_lost).astype(np.uint8)
        else:
            gain_mask = np.zeros_like(bw_new)
            loss_mask = np.zeros_like(bw_lost)

        if np.any(bw_new) or np.any(bw_lost):
            all_masks_empty = False

        area_new_px = int(gain_mask.sum())
        area_lost_px = int(loss_mask.sum())
        area_overlap_px = int(bw_overlap.sum())

        row = {
            "frame_index": k,
            "filename": paths[k].name,
            "is_reference": (k == ref_idx),
            "overlap_w": w_k,
            "overlap_h": h_k,
            "overlap_px": int(w_k * h_k),
            "area_ref_px": prev_area_px,
            "area_mov_px": int(curr_seg.sum()),
            "area_union_px": int(bw_union.sum()),
            "area_new_px": area_new_px,
            "area_lost_px": area_lost_px,
            "area_overlap_px": area_overlap_px,
            "segmentation_method": seg_cfg.get("method"),
            "difference_method": app_cfg.get("difference_method", "abs"),
            "to_ref_transform": T.flatten().tolist(),
        }
        rows.append(row)

        if idx > 0 and save_diagnostics:
            cv2.imencode('.png', (bw_new * 255).astype(np.uint8))[1].tofile(
                str(diff_new_dir / f"{prev_k:04d}_bw_new.png")
            )
            cv2.imencode('.png', (bw_lost * 255).astype(np.uint8))[1].tofile(
                str(diff_lost_dir / f"{prev_k:04d}_bw_lost.png")
            )
            cv2.imencode('.png', (gain_mask * 255).astype(np.uint8))[1].tofile(
                str(diff_gain_dir / f"{prev_k:04d}_bw_gain.png")
            )
            cv2.imencode('.png', (loss_mask * 255).astype(np.uint8))[1].tofile(
                str(diff_loss_dir / f"{prev_k:04d}_bw_loss.png")
            )
        if save_diagnostics:
            cv2.imencode('.png', prev_crop)[1].tofile(
                str(prev_dir / f"{prev_k:04d}_prev.png")
            )
            overlay_color = tuple(app_cfg.get("overlay_mov_color", (255, 0, 255)))
            mov_input = apply_denoising(mov_crop, seg_cfg)
            mov_seg = segment(
                mov_input,
                method=seg_cfg.get("method", "otsu"),
                invert=bool(seg_cfg.get("invert", True)),
                skip_outline=bool(seg_cfg.get("skip_outline", False)),
                manual_thresh=int(seg_cfg.get("manual_thresh", 128)),
                adaptive_block=int(seg_cfg.get("adaptive_block", 51)),
                adaptive_C=int(seg_cfg.get("adaptive_C", 5)),
                local_block=int(seg_cfg.get("local_block", 51)),
                morph_open_radius=(
                    int(seg_cfg["morph_open_radius"])
                    if seg_cfg.get("morph_open_radius") is not None
                    else None
                ),
                morph_close_radius=(
                    int(seg_cfg["morph_close_radius"])
                    if seg_cfg.get("morph_close_radius") is not None
                    else None
                ),
                remove_objects_smaller_px=int(
                    seg_cfg.get("remove_objects_smaller_px", 64)
                ),
                remove_holes_smaller_px=int(
                    seg_cfg.get("remove_holes_smaller_px", 64)
                ),
                use_clahe=bool(seg_cfg.get("use_clahe", False)),
            )
            ov = overlay_outline(mov_crop, mask=mov_seg, color=overlay_color)
            cv2.imencode('.png', ov)[1].tofile(
                str(overlay_dir / f"{k:04d}_overlay_mov.png")
            )

        cv2.imencode('.png', mov_crop)[1].tofile(
            str(mov_dir / f"{k:04d}_mov.png")
        )

        # update previous frame and mask for next iteration
        prev_gray = warped
        # Update the full segmentation mask using the precomputed
        # ``updated_crop`` so that stable regions are carried over to the next
        # iteration regardless of processing direction.
        prev_full_seg[y_k:y_k + h_k, x_k:x_k + w_k] = updated_crop
        prev_k = k

    if all_masks_empty:
        msg = "All segmentation masks were empty"
        logger.error(msg)
        raise ValueError(msg)

    df = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    logger.info("Segmentation complete")
    return df
