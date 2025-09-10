from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from .io_utils import imread_gray, imread_color, ensure_dir
from .registration import register_ecc, register_orb, register_orb_ecc, crop_to_overlap, preprocess
from .segmentation import segment
from .background import normalize_background, estimate_temporal_background
import logging
import re

logger = logging.getLogger(__name__)

def overlay_outline(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours, -1, (0,255,0), 1)
    return color

def analyze_sequence(paths: List[Path], reg_cfg: dict, seg_cfg: dict, app_cfg: dict, out_dir: Path) -> pd.DataFrame:
    norm = bool(app_cfg.get("normalize", True))
    scale_minmax = app_cfg.get("scale_minmax")
    imgs_gray = [imread_gray(p, normalize=norm, scale_minmax=scale_minmax) for p in paths]
    H, W = imgs_gray[0].shape[:2]

    logger.info("Loaded %d images with dimensions %dx%d", len(imgs_gray), H, W)

    # Determine processing order and starting reference frame
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

    # Optionally subtract a temporal background using early frames
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

    # Prepare outputs
    ensure_dir(out_dir)
    reg_dir = out_dir / "registered"; ensure_dir(reg_dir)
    bw_dir = out_dir / "binary"; ensure_dir(bw_dir)
    diff_dir = out_dir / "diff"; ensure_dir(diff_dir)
    overlay_dir = out_dir / "overlay"; ensure_dir(overlay_dir)

    rows = []

    # Transformation chain mapping each frame back to the chosen
    # starting reference.  Each step registers the current frame to the
    # previously processed original frame and composes the transforms.
    transforms: Dict[int, np.ndarray] = {ref_idx: np.eye(3, dtype=np.float32)}

    # Initialize mask and bounding box covering the full image.  The
    # bbox is updated after each registration to track the common area
    # across all frames.
    if initial_radius > 0:
        ecc_mask = np.zeros((H, W), dtype=np.uint8)
        cx, cy = W // 2, H // 2
        x0 = max(cx - initial_radius, 0)
        y0 = max(cy - initial_radius, 0)
        x1 = min(cx + initial_radius, W)
        y1 = min(cy + initial_radius, H)
        ecc_mask[y0:y1, x0:x1] = 255
        bbox_x, bbox_y, bbox_w, bbox_h = x0, y0, x1 - x0, y1 - y0
    else:
        ecc_mask = np.ones((H, W), dtype=np.uint8) * 255
        bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, W, H

    # Iterate through frames in the chosen order. Each step registers
    # the current frame to the previously processed original frame to
    # avoid compounding warps.
    for idx, k in enumerate(ordered_indices):
        g_full = imgs_norm[k]

        logger.debug("Frame %d: processing", k)

        if idx == 0:
            logger.debug("Frame %d: reference frame, no registration", k)
            # Starting reference frame: no registration needed.
            ref_gray = g_full
            warped = ref_gray.copy()
            valid_mask = ecc_mask.copy()
        else:
            prev_k = ordered_indices[idx - 1]
            prev_full = imgs_norm[prev_k]
            ref_gray = prev_full

            method = reg_cfg.get("method", "ECC").upper()
            logger.debug("Frame %d: registration method %s", k, method)
            if method == "ORB":
                success, W_step, warped, valid_mask, fb, _, _ = register_orb(
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
                if not success:
                    logger.warning("Registration failed at frame %d", k)
                    logger.debug("Frame %d: registration failed", k)
                    continue
                logger.debug("Frame %d: registration succeeded", k)
            elif method == "ORB+ECC":
                success, W_step, warped, valid_mask, _, _ = register_orb_ecc(
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
                    mask=ecc_mask if reg_cfg.get("use_masked_ecc", True) else None,
                )
                if not success:
                    logger.warning("Registration failed at frame %d", k)
                    logger.debug("Frame %d: registration failed", k)
                    continue
                logger.debug("Frame %d: registration succeeded", k)
            else:
                success, W_step, warped, valid_mask = register_ecc(
                    ref_gray,
                    g_full,
                    model=reg_cfg.get("model", "affine"),
                    max_iters=int(reg_cfg.get("max_iters", 1000)),
                    eps=float(reg_cfg.get("eps", 1e-6)),
                    mask=ecc_mask if reg_cfg.get("use_masked_ecc", True) else None,
                )
                if not success:
                    logger.warning("Registration failed at frame %d", k)
                    logger.debug("Frame %d: registration failed", k)
                    continue
                logger.debug("Frame %d: registration succeeded", k)

            W_h = W_step if W_step.shape == (3, 3) else np.vstack([W_step, [0, 0, 1]])
            transforms[k] = transforms[prev_k] @ W_h

        # Restrict valid region to the accumulated mask
        valid_mask = cv2.bitwise_and(valid_mask, ecc_mask)

        # Segment the reference frame corresponding to this step.
        logger.debug("Frame %d: starting segmentation", k)
        bw_ref = segment(ref_gray,
                         method=seg_cfg.get("method", "otsu"),
                         invert=bool(seg_cfg.get("invert", True)),
                         manual_thresh=int(seg_cfg.get("manual_thresh", 128)),
                         adaptive_block=int(seg_cfg.get("adaptive_block", 51)),
                         adaptive_C=int(seg_cfg.get("adaptive_C", 5)),
                         local_block=int(seg_cfg.get("local_block", 51)),
                         morph_open_radius=int(seg_cfg.get("morph_open_radius", 2)),
                         morph_close_radius=int(seg_cfg.get("morph_close_radius", 2)),
                         remove_objects_smaller_px=int(seg_cfg.get("remove_objects_smaller_px", 64)),
                         remove_holes_smaller_px=int(seg_cfg.get("remove_holes_smaller_px", 64)))

        x, y, w, h = crop_to_overlap(valid_mask)
        if w == 0 or h == 0:
            logger.debug("Frame %d: empty overlap region", k)
            continue

        ref_crop = ref_gray[y:y + h, x:x + w]
        mov_crop = warped[y:y + h, x:x + w]
        bw_ref_crop = bw_ref[y:y + h, x:x + w]

        # Segmentation on diff or raw
        if app_cfg.get("use_difference_for_seg", False):
            diff_abs = cv2.absdiff(ref_crop, mov_crop)
            seg_img = diff_abs
        else:
            seg_img = mov_crop

        bw_mov = segment(seg_img,
                         method=seg_cfg.get("method", "otsu"),
                         invert=bool(seg_cfg.get("invert", True)),
                         manual_thresh=int(seg_cfg.get("manual_thresh", 128)),
                         adaptive_block=int(seg_cfg.get("adaptive_block", 51)),
                         adaptive_C=int(seg_cfg.get("adaptive_C", 5)),
                         local_block=int(seg_cfg.get("local_block", 51)),
                         morph_open_radius=int(seg_cfg.get("morph_open_radius", 2)),
                         morph_close_radius=int(seg_cfg.get("morph_close_radius", 2)),
                         remove_objects_smaller_px=int(seg_cfg.get("remove_objects_smaller_px", 64)),
                         remove_holes_smaller_px=int(seg_cfg.get("remove_holes_smaller_px", 64)))

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
            # Flattened homography mapping this frame back to the
            # starting reference frame.
            "to_ref_transform": transforms.get(k, np.eye(3)).flatten().tolist(),
        }
        rows.append(row)
        logger.debug(
            "Frame %d: segmentation complete overlap_px=%d area_ref=%d area_mov=%d area_new=%d area_lost=%d",
            k,
            row["overlap_px"],
            row["area_ref_px"],
            row["area_mov_px"],
            row["area_new_px"],
            row["area_lost_px"],
        )

        if app_cfg.get("save_intermediates", True):
            cv2.imencode('.png', ref_crop)[1].tofile(str(reg_dir / f"{k:04d}_ref.png"))
            cv2.imencode('.png', mov_crop)[1].tofile(str(reg_dir / f"{k:04d}_mov.png"))
            cv2.imencode('.png', (bw_mov * 255).astype(np.uint8))[1].tofile(str(bw_dir / f"{k:04d}_bw_mov.png"))
            cv2.imencode('.png', (bw_ref_crop * 255).astype(np.uint8))[1].tofile(str(bw_dir / f"{k:04d}_bw_ref.png"))
            cv2.imencode('.png', (bw_overlap * 255).astype(np.uint8))[1].tofile(str(bw_dir / f"{k:04d}_bw_overlap.png"))
            cv2.imencode('.png', (bw_new * 255).astype(np.uint8))[1].tofile(str(diff_dir / f"{k:04d}_bw_new.png"))
            cv2.imencode('.png', (bw_lost * 255).astype(np.uint8))[1].tofile(str(diff_dir / f"{k:04d}_bw_lost.png"))
            ov = overlay_outline(mov_crop, bw_mov)
            cv2.imencode('.png', ov)[1].tofile(str(overlay_dir / f"{k:04d}_overlay_mov.png"))
            saved_paths = [
                reg_dir / f"{k:04d}_ref.png",
                reg_dir / f"{k:04d}_mov.png",
                bw_dir / f"{k:04d}_bw_mov.png",
                bw_dir / f"{k:04d}_bw_ref.png",
                bw_dir / f"{k:04d}_bw_overlap.png",
                diff_dir / f"{k:04d}_bw_new.png",
                diff_dir / f"{k:04d}_bw_lost.png",
                overlay_dir / f"{k:04d}_overlay_mov.png",
            ]
            logger.debug("Frame %d: saved intermediates %s", k, [str(p) for p in saved_paths])

        # Update mask and bbox for next iteration, allowing the region to
        # shrink or grow according to ``growth_factor``.
        w2 = max(1, min(int(w * growth_factor), W))
        h2 = max(1, min(int(h * growth_factor), H))
        x2 = max(0, x - (w2 - w) // 2)
        y2 = max(0, y - (h2 - h) // 2)
        x2 = min(x2, W - w2)
        y2 = min(y2, H - h2)
        ecc_mask = np.zeros((H, W), dtype=np.uint8)
        ecc_mask[y2:y2 + h2, x2:x2 + w2] = 255
        bbox_x, bbox_y, bbox_w, bbox_h = x2, y2, w2, h2
        logger.debug(
            "Frame %d: bbox updated to x=%d y=%d w=%d h=%d",
            k,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
        )

    df = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    summary_path = out_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info("Segmentation complete; summary saved to %s", summary_path)
    return df
