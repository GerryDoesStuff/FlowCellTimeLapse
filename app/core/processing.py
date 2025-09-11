from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Optional, Dict
from .io_utils import imread_gray, imread_color, ensure_dir
from .registration import register_ecc, register_orb, register_orb_ecc, crop_to_overlap, preprocess
from .segmentation import segment
from .difference import compute_difference
from .background import normalize_background, estimate_temporal_background
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

    ensure_dir(out_dir)
    reg_dir = out_dir / "registered"; ensure_dir(reg_dir)

    bw_dir = out_dir / "binary"; ensure_dir(bw_dir)
    bw_mov_dir = bw_dir / "mov"; ensure_dir(bw_mov_dir)
    bw_prev_dir = bw_dir / "prev"; ensure_dir(bw_prev_dir)
    bw_overlap_dir = bw_dir / "overlap"; ensure_dir(bw_overlap_dir)
    bw_empty_dir = bw_dir / "empty"; ensure_dir(bw_empty_dir)

    diff_dir = out_dir / "diff"; ensure_dir(diff_dir)
    diff_diff_dir = diff_dir / "diff"; ensure_dir(diff_diff_dir)
    diff_new_dir = diff_dir / "new"; ensure_dir(diff_new_dir)
    diff_lost_dir = diff_dir / "lost"; ensure_dir(diff_lost_dir)

    overlay_dir = out_dir / "overlay"; ensure_dir(overlay_dir)

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

    final_mask = (global_mask > 0).astype(np.uint8)
    for idx, k in enumerate(ordered_indices[1:], start=1):
        vm = valid_masks.get(k)
        if vm is None:
            continue
        prev_k = ordered_indices[idx - 1]
        T_prev = step_transforms.get(prev_k, np.eye(3, dtype=np.float32))
        vm_ref = cv2.warpPerspective(vm, T_prev, (W, H))
        final_mask = cv2.bitwise_and(final_mask, vm_ref)

    crop_rect = crop_to_overlap(final_mask)
    crop_rects: Dict[int, tuple[int, int, int, int]] = {k: crop_rect for k in ordered_indices}
    x_f, y_f, w_f, h_f = crop_rect
    overlap_area = w_f * h_f
    min_overlap = int(0.01 * H * W)
    if overlap_area < min_overlap:
        msg = f"Final overlap area {overlap_area} below 1% threshold {min_overlap}"
        logger.error(msg)
        raise ValueError(msg)
    if app_cfg.get("save_final_mask", False):
        cv2.imencode('.png', (final_mask * 255).astype(np.uint8))[1].tofile(
            str(out_dir / "final_mask.png")
        )

    # Phase 2: segmentation using per-frame masks
    def _save_mask(
        idx: int,
        mask: np.ndarray,
        x_off: int,
        y_off: int,
        *,
        suffix: str = "",
        overlay: bool = True,
    ) -> None:
        if not app_cfg.get("save_masks", False):
            return
        h, w = mask.shape[:2]
        full_mask = np.zeros_like(imgs_gray[idx], dtype=mask.dtype)
        full_mask[y_off : y_off + h, x_off : x_off + w] = mask
        cv2.imencode(".png", (full_mask * 255).astype(np.uint8))[1].tofile(
            str(out_dir / f"mask_{idx:04d}{suffix}.png")
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
                str(out_dir / f"mask_{idx:04d}_overlay{suffix}.png")
            )

    ref_gray = registered_frames[ref_idx]
    bw_ref = segment(
        ref_gray,
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
    )

    # store previous frame, mask, and index for iterative segmentation
    prev_gray = ref_gray
    prev_bw = bw_ref
    prev_k = ref_idx

    ecc_mask = None
    all_masks_empty = True

    for idx, k in enumerate(ordered_indices):
        logger.debug("Frame %d: segmentation phase", k)
        x_k, y_k, w_k, h_k = crop_rects.get(k, (0, 0, W, H))
        prev_crop = prev_gray[y_k:y_k + h_k, x_k:x_k + w_k]
        prev_bw_crop = prev_bw[y_k:y_k + h_k, x_k:x_k + w_k]
        T = step_transforms.get(k, np.eye(3, dtype=np.float32))
        warped = registered_frames.get(k)
        if warped is None:
            warped = cv2.warpPerspective(imgs_norm[k], T, (W, H))
            registered_frames[k] = warped
        mov_crop = warped[y_k:y_k + h_k, x_k:x_k + w_k]
        seg_img = None
        bw_diff = None
        if idx > 0:
            seg_img = compute_difference(
                prev_crop, mov_crop, method=app_cfg.get("difference_method", "abs")
            )
            bw_diff = segment(
                seg_img,
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
                remove_objects_smaller_px=int(seg_cfg.get("remove_objects_smaller_px", 64)),
                remove_holes_smaller_px=int(seg_cfg.get("remove_holes_smaller_px", 64)),
            )
            if app_cfg.get("save_intermediates", True):
                cv2.imencode(".png", seg_img)[1].tofile(
                    str(diff_diff_dir / f"{k:04d}_diff.png")
                )
                cv2.imencode(".png", (bw_diff * 255).astype(np.uint8))[1].tofile(
                    str(diff_diff_dir / f"{k:04d}_bw_diff.png")
                )
        bw_reg = segment(
            mov_crop,
            method=seg_cfg.get("method", "otsu"),
            invert=bool(seg_cfg.get("invert", True)),
            skip_outline=bool(seg_cfg.get("skip_outline", False)),
            use_diff=False,
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
        )

        seg_mask = bw_diff if (bw_diff is not None) else bw_reg

        _save_mask(k, bw_reg, x_k, y_k, suffix="_registered")
        if bw_diff is not None:
            _save_mask(k, bw_diff, x_k, y_k, suffix="_difference", overlay=False)

        if not np.any(seg_mask):
            logger.warning(
                "Frame %d: segmentation mask is empty; skipping ecc_mask update", k
            )
            cv2.imencode(".png", (seg_mask * 255).astype(np.uint8))[1].tofile(
                str(bw_empty_dir / f"{k:04d}_bw_mov_empty.png")
            )
        else:
            all_masks_empty = False
            ecc_mask = seg_mask.copy()

        bw_overlap = (prev_bw_crop & seg_mask).astype(np.uint8)
        bw_union = (prev_bw_crop | seg_mask).astype(np.uint8)
        bw_new = (seg_mask & (~prev_bw_crop)).astype(np.uint8)
        bw_lost = (prev_bw_crop & (~seg_mask)).astype(np.uint8)

        row = {
            "frame_index": k,
            "filename": paths[k].name,
            "is_reference": (k == ref_idx),
            "overlap_w": w_k,
            "overlap_h": h_k,
            "overlap_px": int(w_k * h_k),
            "area_ref_px": int(prev_bw_crop.sum()),
            "area_mov_px": int(seg_mask.sum()),
            "area_union_px": int(bw_union.sum()),
            "area_new_px": int(bw_new.sum()),
            "area_lost_px": int(bw_lost.sum()),
            "to_ref_transform": T.flatten().tolist(),
        }
        rows.append(row)

        if app_cfg.get("save_intermediates", True):
            cv2.imencode('.png', prev_crop)[1].tofile(
                str(reg_dir / f"{prev_k:04d}_prev.png")
            )
            cv2.imencode('.png', mov_crop)[1].tofile(
                str(reg_dir / f"{k:04d}_mov.png")
            )
            cv2.imencode('.png', (seg_mask * 255).astype(np.uint8))[1].tofile(
                str(bw_mov_dir / f"{k:04d}_bw_mov.png")
            )
            if bw_diff is not None:
                cv2.imencode('.png', (bw_reg * 255).astype(np.uint8))[1].tofile(
                    str(bw_dir / f"{k:04d}_bw_reg.png")
                )
                cv2.imencode('.png', (bw_diff * 255).astype(np.uint8))[1].tofile(
                    str(bw_dir / f"{k:04d}_bw_diff.png")
                )
            cv2.imencode('.png', (prev_bw_crop * 255).astype(np.uint8))[1].tofile(
                str(bw_prev_dir / f"{prev_k:04d}_bw_prev.png")
            )
            cv2.imencode('.png', (bw_overlap * 255).astype(np.uint8))[1].tofile(
                str(bw_overlap_dir / f"{prev_k:04d}_bw_overlap.png")
            )
            if idx > 0:
                cv2.imencode('.png', (bw_new * 255).astype(np.uint8))[1].tofile(
                    str(diff_new_dir / f"{prev_k:04d}_bw_new.png")
                )
                cv2.imencode('.png', (bw_lost * 255).astype(np.uint8))[1].tofile(
                    str(diff_lost_dir / f"{prev_k:04d}_bw_lost.png")
                )
                new_color = tuple(app_cfg.get("overlay_new_color", (0, 255, 0)))
                lost_color = tuple(app_cfg.get("overlay_lost_color", (0, 0, 255)))
                ov = overlay_outline(
                    mov_crop,
                    new_mask=bw_new,
                    lost_mask=bw_lost,
                    new_color=new_color,
                    lost_color=lost_color,
                )
                cv2.imencode('.png', ov)[1].tofile(
                    str(overlay_dir / f"{prev_k:04d}_overlay_mov.png")
                )

        # update previous frame and mask for next iteration
        prev_gray = warped
        prev_bw = np.zeros_like(prev_bw)
        prev_bw[y_k:y_k + h_k, x_k:x_k + w_k] = seg_mask
        prev_k = k

    if all_masks_empty:
        msg = "All segmentation masks were empty"
        logger.error(msg)
        raise ValueError(msg)

    df = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    summary_path = out_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info("Segmentation complete; summary saved to %s", summary_path)
    return df
