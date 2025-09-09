from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from .io_utils import imread_gray, imread_color, ensure_dir
from .registration import register_ecc, register_orb, crop_to_overlap, preprocess
from .segmentation import segment
from .background import normalize_background, estimate_temporal_background

def overlay_outline(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours, -1, (0,255,0), 1)
    return color

def analyze_sequence(paths: List[Path], reg_cfg: dict, seg_cfg: dict, app_cfg: dict, out_dir: Path) -> pd.DataFrame:
    imgs_gray = [imread_gray(p) for p in paths]
    H, W = imgs_gray[0].shape[:2]

    # Determine starting reference frame based on user selection
    ref_choice = app_cfg.get("reference_choice", "last")
    if ref_choice == "first":
        ref_idx = 0
    elif ref_choice == "middle":
        ref_idx = len(paths) // 2
    elif ref_choice == "custom":
        # Clamp custom index to valid range
        ref_idx = int(app_cfg.get("custom_ref_index", 0))
        ref_idx = max(0, min(len(paths) - 1, ref_idx))
    else:
        # Default to using the last frame
        ref_idx = len(paths) - 1

    # Background normalization using early frames
    bg = estimate_temporal_background(imgs_gray, n_early=5)
    imgs_norm = [normalize_background(g, bg) for g in imgs_gray]
    gauss_sigma = float(reg_cfg.get("gauss_blur_sigma", 1.0))
    clahe_clip = float(reg_cfg.get("clahe_clip", 2.0))
    clahe_grid = int(reg_cfg.get("clahe_grid", 8))
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
    # previous original frame and composes the transforms.
    transforms: Dict[int, np.ndarray] = {ref_idx: np.eye(3, dtype=np.float32)}

    # Initialize mask and bounding box covering the full image.  The
    # bbox is updated after each registration to track the common area
    # across all frames.
    ecc_mask = np.ones((H, W), dtype=np.uint8) * 255
    bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, W, H

    # Iterate backwards from the reference frame so that at step k the
    # registration target is the original frame at k+1 (i.e. the next
    # later frame).  This avoids repeatedly warping already warped
    # images and yields a transformation chain relative to the initial
    # reference.
    for k in range(ref_idx, -1, -1):
        g_full = imgs_norm[k]
        g_norm = g_full[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w]

        if k == ref_idx:
            # Starting reference frame: no registration needed.
            ref_gray = g_norm
            warped = ref_gray.copy()
            valid_mask = np.ones_like(ref_gray, dtype=np.uint8) * 255
        else:
            # Use the original (unwarped) previous frame as the
            # registration target for the current frame.
            prev_full = imgs_norm[k + 1]
            ref_gray = prev_full[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w]

            if reg_cfg.get("method", "ECC").upper() == "ORB":
                W_step, warped, valid_mask = register_orb(ref_gray, g_norm, model=reg_cfg.get("model", "homography"))
            else:
                W_step, warped, valid_mask = register_ecc(
                    ref_gray,
                    g_norm,
                    model=reg_cfg.get("model", "affine"),
                    max_iters=int(reg_cfg.get("max_iters", 1000)),
                    eps=float(reg_cfg.get("eps", 1e-6)),
                    mask=ecc_mask if reg_cfg.get("use_masked_ecc", True) else None,
                )

            # Compose the step with the cumulative transform so we can
            # later map this frame back to the starting reference.
            W_h = W_step if W_step.shape == (3, 3) else np.vstack([W_step, [0, 0, 1]])
            transforms[k] = transforms[k + 1] @ W_h

        # Segment the reference frame corresponding to this step.
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

        # Update mask and bbox for next iteration. The reference frame
        # for the next step will be loaded fresh from disk, so only the
        # ROI and mask need to be carried forward.
        ecc_mask = valid_mask[y:y + h, x:x + w]
        bbox_x += x
        bbox_y += y
        bbox_w = w
        bbox_h = h

    df = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df
