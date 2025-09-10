from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Optional, Dict
from .io_utils import imread_gray, imread_color, ensure_dir
from .registration import register_ecc, register_orb, register_orb_ecc, crop_to_overlap, preprocess
from .segmentation import segment
from .background import normalize_background, estimate_temporal_background
import logging
import re

logger = logging.getLogger(__name__)


def overlay_outline(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours, -1, (0, 255, 0), 1)
    return color


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
    initial_radius = int(reg_cfg.get("initial_radius", min(H, W) // 2))
    growth_factor = float(reg_cfg.get("growth_factor", 1.0))
    logger.info(
        "Preprocessing parameters: gauss_sigma=%.2f clahe_clip=%.2f clahe_grid=%d initial_radius=%d growth_factor=%.2f",
        gauss_sigma,
        clahe_clip,
        clahe_grid,
        initial_radius,
        growth_factor,
    )
    imgs_norm = [preprocess(g, gauss_sigma, clahe_clip, clahe_grid) for g in imgs_norm]

    ensure_dir(out_dir)
    reg_dir = out_dir / "registered"; ensure_dir(reg_dir)
    bw_dir = out_dir / "binary"; ensure_dir(bw_dir)
    diff_dir = out_dir / "diff"; ensure_dir(diff_dir)
    overlay_dir = out_dir / "overlay"; ensure_dir(overlay_dir)

    rows: List[Dict] = []

    transforms: Dict[int, np.ndarray] = {ref_idx: np.eye(3, dtype=np.float32)}

    if initial_radius > 0:
        global_mask = np.zeros((H, W), dtype=np.uint8)
        cx, cy = W // 2, H // 2
        x0 = max(cx - initial_radius, 0)
        y0 = max(cy - initial_radius, 0)
        x1 = min(cx + initial_radius, W)
        y1 = min(cy + initial_radius, H)
        global_mask[y0:y1, x0:x1] = 255
    else:
        global_mask = np.ones((H, W), dtype=np.uint8) * 255

    # Phase 1: register frames and accumulate global mask
    for idx, k in enumerate(ordered_indices):
        g_full = imgs_norm[k]
        logger.debug("Frame %d: registration phase", k)
        if idx == 0:
            valid_mask = global_mask.copy()
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
                transforms[k] = transforms[prev_k]
                continue
            W_h = W_step if W_step.shape == (3, 3) else np.vstack([W_step, [0, 0, 1]])
            transforms[k] = transforms[prev_k] @ W_h

        new_mask = cv2.bitwise_and(global_mask, valid_mask)
        overlap_area = int(new_mask.sum())
        min_overlap = int(0.01 * H * W)
        if overlap_area < min_overlap and idx > 0:
            logger.warning(
                "Frame %d: overlap area %d below 1%% threshold %d; keeping previous mask",
                k,
                overlap_area,
                min_overlap,
            )
        else:
            global_mask = new_mask

    x, y, w, h = crop_to_overlap(global_mask)
    if w == 0 or h == 0:
        logger.warning("Final overlap mask is empty; using full frame")
        x, y, w, h = 0, 0, W, H
        global_mask[:, :] = 255

    min_area = int(0.8 * W * H)
    area = w * h
    if area < min_area:
        logger.warning(
            "Final overlap area %d below 80%% threshold %d; using full frame",
            area,
            min_area,
        )
        x, y, w, h = 0, 0, W, H
        global_mask[:, :] = 255

    if app_cfg.get("save_final_mask", False):
        cv2.imencode('.png', global_mask)[1].tofile(str(out_dir / "final_mask.png"))

    # Phase 2: segmentation using the final mask
    ref_gray = imgs_norm[ref_idx]
    bw_ref = segment(
        ref_gray,
        method=seg_cfg.get("method", "otsu"),
        invert=bool(seg_cfg.get("invert", True)),
        manual_thresh=int(seg_cfg.get("manual_thresh", 128)),
        adaptive_block=int(seg_cfg.get("adaptive_block", 51)),
        adaptive_C=int(seg_cfg.get("adaptive_C", 5)),
        local_block=int(seg_cfg.get("local_block", 51)),
        morph_open_radius=int(seg_cfg.get("morph_open_radius", 2)),
        morph_close_radius=int(seg_cfg.get("morph_close_radius", 2)),
        remove_objects_smaller_px=int(seg_cfg.get("remove_objects_smaller_px", 64)),
        remove_holes_smaller_px=int(seg_cfg.get("remove_holes_smaller_px", 64)),
    )
    ref_crop = ref_gray[y:y + h, x:x + w]
    bw_ref_crop = bw_ref[y:y + h, x:x + w]

    ecc_mask = None
    if not np.any(bw_ref_crop):
        logger.warning("Reference segmentation mask is empty; skipping ecc_mask update")
        cv2.imencode('.png', (bw_ref_crop * 255).astype(np.uint8))[1].tofile(
            str(bw_dir / f"{ref_idx:04d}_bw_ref_empty.png")
        )
    else:
        ecc_mask = bw_ref_crop.copy()

    for k in ordered_indices:
        logger.debug("Frame %d: segmentation phase", k)
        warped = cv2.warpPerspective(imgs_norm[k], transforms[k], (W, H))
        mov_crop = warped[y:y + h, x:x + w]
        if app_cfg.get("use_difference_for_seg", False):
            seg_img = cv2.absdiff(ref_crop, mov_crop)
        else:
            seg_img = mov_crop
        bw_mov = segment(
            seg_img,
            method=seg_cfg.get("method", "otsu"),
            invert=bool(seg_cfg.get("invert", True)),
            manual_thresh=int(seg_cfg.get("manual_thresh", 128)),
            adaptive_block=int(seg_cfg.get("adaptive_block", 51)),
            adaptive_C=int(seg_cfg.get("adaptive_C", 5)),
            local_block=int(seg_cfg.get("local_block", 51)),
            morph_open_radius=int(seg_cfg.get("morph_open_radius", 2)),
            morph_close_radius=int(seg_cfg.get("morph_close_radius", 2)),
            remove_objects_smaller_px=int(seg_cfg.get("remove_objects_smaller_px", 64)),
            remove_holes_smaller_px=int(seg_cfg.get("remove_holes_smaller_px", 64)),
        )
        if not np.any(bw_mov):
            logger.warning(
                "Frame %d: segmentation mask is empty; skipping ecc_mask update", k
            )
            cv2.imencode('.png', (bw_mov * 255).astype(np.uint8))[1].tofile(
                str(bw_dir / f"{k:04d}_bw_mov_empty.png")
            )
        else:
            ecc_mask = bw_mov.copy()

        bw_overlap = (bw_ref_crop & bw_mov).astype(np.uint8)
        bw_union = (bw_ref_crop | bw_mov).astype(np.uint8)
        bw_new = (bw_mov & (~bw_ref_crop)).astype(np.uint8)
        bw_lost = (bw_ref_crop & (~bw_mov)).astype(np.uint8)

        row = {
            "frame_index": k,
            "filename": paths[k].name,
            "is_reference": (k == ref_idx),
            "overlap_w": w,
            "overlap_h": h,
            "overlap_px": int(w * h),
            "area_ref_px": int(bw_ref_crop.sum()),
            "area_mov_px": int(bw_mov.sum()),
            "area_union_px": int(bw_union.sum()),
            "area_new_px": int(bw_new.sum()),
            "area_lost_px": int(bw_lost.sum()),
            "to_ref_transform": transforms.get(k, np.eye(3)).flatten().tolist(),
        }
        rows.append(row)

        if app_cfg.get("save_intermediates", True):
            cv2.imencode('.png', ref_crop)[1].tofile(str(reg_dir / f"{k:04d}_ref.png"))
            cv2.imencode('.png', mov_crop)[1].tofile(str(reg_dir / f"{k:04d}_mov.png"))
            cv2.imencode('.png', (bw_mov * 255).astype(np.uint8))[1].tofile(
                str(bw_dir / f"{k:04d}_bw_mov.png")
            )
            cv2.imencode('.png', (bw_ref_crop * 255).astype(np.uint8))[1].tofile(
                str(bw_dir / f"{k:04d}_bw_ref.png")
            )
            cv2.imencode('.png', (bw_overlap * 255).astype(np.uint8))[1].tofile(
                str(bw_dir / f"{k:04d}_bw_overlap.png")
            )
            cv2.imencode('.png', (bw_new * 255).astype(np.uint8))[1].tofile(
                str(diff_dir / f"{k:04d}_bw_new.png")
            )
            cv2.imencode('.png', (bw_lost * 255).astype(np.uint8))[1].tofile(
                str(diff_dir / f"{k:04d}_bw_lost.png")
            )
            ov = overlay_outline(mov_crop, bw_mov)
            cv2.imencode('.png', ov)[1].tofile(str(overlay_dir / f"{k:04d}_overlay_mov.png"))

    df = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    summary_path = out_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info("Segmentation complete; summary saved to %s", summary_path)
    return df
