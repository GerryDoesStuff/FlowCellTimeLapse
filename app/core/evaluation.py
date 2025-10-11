from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd


def _index_from_name(path: Path) -> int:
    """Extract the leading frame index from ``path``."""
    m = re.match(r"(\d+)", path.stem)
    if not m:
        raise ValueError(f"Unexpected filename {path.name}")
    return int(m.group(1))


def _read_mask(path: Path | None) -> np.ndarray | None:
    """Read a mask image if ``path`` exists.

    Parameters
    ----------
    path : Path | None
        Image path. When ``None`` or missing, ``None`` is returned.
    """
    if path is None or not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read {path}")
    return mask


def write_shape_properties(
    mask: np.ndarray,
    csv_path: Path | str,
    *,
    frame_index: int,
    frame_name: str | None = None,
) -> None:
    """Append connected component properties of ``mask`` to ``csv_path``.

    Parameters
    ----------
    mask:
        Binary mask where non-zero pixels represent the components of
        interest. Values are interpreted in a binary fashion.
    csv_path:
        Destination CSV file. Created if it does not yet exist. The
        resulting table contains one row per connected component with the
        columns ``area_px``, ``centroid_x``, ``centroid_y``, ``bbox_left``,
        ``bbox_top``, ``bbox_width`` and ``bbox_height`` together with the
        provided metadata columns.
    frame_index:
        Index of the frame from which ``mask`` originates. Stored with every
        connected component row to simplify downstream filtering.
    frame_name:
        Optional human-readable frame identifier. When provided this value is
        stored alongside the numeric index.
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin)

    rows: list[dict[str, Any]] = []
    for lbl in range(1, num_labels):  # Skip background
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        cx, cy = centroids[lbl]
        row: dict[str, Any] = {
            "frame_index": frame_index,
            "area_px": area,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "bbox_left": x,
            "bbox_top": y,
            "bbox_width": w,
            "bbox_height": h,
        }
        if frame_name is not None:
            row["frame_name"] = frame_name
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def evaluate_diff_masks(diff_dir: Path) -> pd.DataFrame:
    """Evaluate binary difference masks.

    This function scans ``diff_dir`` for ``bw``, ``gain`` and ``loss``
    subfolders produced by :func:`app.core.processing.analyze_sequence`. When
    the older ``new``/``lost`` folders are present they are used instead. For
    each frame interval it loads the corresponding masks and computes simple
    area metrics.

    Parameters
    ----------
    diff_dir : Path
        Directory containing ``bw`` and ``gain``/``loss`` subdirectories.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``frame_index``, ``area_diff_px``, ``area_new_px``,
        ``area_lost_px`` and ``net_new_px``.
    """
    diff_dir = Path(diff_dir)
    bw_dir = diff_dir / "bw"
    new_dir = diff_dir / "new"
    lost_dir = diff_dir / "lost"
    gain_dir = diff_dir / "gain"
    loss_dir = diff_dir / "loss"

    bw_map: Dict[int, Path] = {}
    new_map: Dict[int, Path] = {}
    lost_map: Dict[int, Path] = {}

    if bw_dir.exists():
        for p in bw_dir.glob("*.png"):
            bw_map[_index_from_name(p)] = p

    if gain_dir.exists():
        for p in gain_dir.glob("*.png"):
            new_map[_index_from_name(p)] = p
    elif new_dir.exists():
        for p in new_dir.glob("*.png"):
            new_map[_index_from_name(p)] = p

    if loss_dir.exists():
        for p in loss_dir.glob("*.png"):
            lost_map[_index_from_name(p)] = p
    elif lost_dir.exists():
        for p in lost_dir.glob("*.png"):
            lost_map[_index_from_name(p)] = p

    frame_indices = sorted(set(new_map) | set(lost_map))
    rows: list[Dict[str, Any]] = []
    for idx in frame_indices:
        new_mask = _read_mask(new_map.get(idx))
        lost_mask = _read_mask(lost_map.get(idx))
        # ``bw`` masks are indexed by the moving frame (idx+1)
        diff_mask = _read_mask(bw_map.get(idx + 1))
        if diff_mask is None:
            diff_mask = _read_mask(bw_map.get(idx))

        area_new = int(np.count_nonzero(new_mask)) if new_mask is not None else 0
        area_lost = int(np.count_nonzero(lost_mask)) if lost_mask is not None else 0
        area_diff = int(np.count_nonzero(diff_mask)) if diff_mask is not None else 0

        rows.append(
            {
                "frame_index": idx,
                "area_diff_px": area_diff,
                "area_new_px": area_new,
                "area_lost_px": area_lost,
                "net_new_px": area_new - area_lost,
            }
        )

    df = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    return df
