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


def evaluate_diff_masks(
    diff_dir: Path,
    *,
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Evaluate binary difference masks.

    This function scans ``diff_dir`` for ``bw``, ``new`` and ``lost`` subfolders
    produced by :func:`app.core.processing.analyze_sequence`. For each frame
    interval it loads the corresponding masks and computes simple area metrics.

    Parameters
    ----------
    diff_dir : Path
        Directory containing ``bw``, ``new`` and ``lost`` subdirectories.
    csv_path : Path | str | None, optional
        When provided, the resulting :class:`pandas.DataFrame` is written to this
        CSV file. Relative paths are resolved inside ``diff_dir``.

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

    bw_map: Dict[int, Path] = {}
    new_map: Dict[int, Path] = {}
    lost_map: Dict[int, Path] = {}

    if bw_dir.exists():
        for p in bw_dir.glob("*.png"):
            bw_map[_index_from_name(p)] = p
    if new_dir.exists():
        for p in new_dir.glob("*.png"):
            new_map[_index_from_name(p)] = p
    if lost_dir.exists():
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

    if csv_path is not None:
        csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            csv_path = diff_dir / csv_path
        df.to_csv(csv_path, index=False)

    return df
