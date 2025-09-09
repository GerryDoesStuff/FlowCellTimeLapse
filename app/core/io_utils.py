from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import os
import re
from datetime import datetime

SUPPORTED_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}

def discover_images(folder: Path, numeric_sort: bool=True) -> list[Path]:
    paths = []
    for ext in SUPPORTED_EXTS:
        paths.extend(sorted(folder.glob(f"*{ext}")))
    if not paths:
        for ext in SUPPORTED_EXTS:
            paths.extend(sorted(folder.rglob(f"*{ext}")))
    if numeric_sort:
        def key(p: Path):
            nums = re.findall(r"\d+", p.name)
            return tuple(int(n) for n in nums) if nums else (float("inf"), p.name)
        paths = sorted(paths, key=key)
    else:
        paths = sorted(paths, key=lambda p: p.name)
    return paths

def imread_gray(path: Path, normalize: bool = True,
                scale_minmax: tuple[int, int] | None = None) -> np.ndarray:
    """Read an image and return a grayscale ``uint8`` array.

    Parameters
    ----------
    path: Path
        Image file to read.
    normalize: bool, default True
        If ``True`` the image is scaled to ``uint8``. When ``False`` the
        original dtype/range is preserved.
    scale_minmax: tuple[int, int] | None, optional
        When provided, these values are treated as the global minimum and
        maximum across all frames.  The image is clipped to this range and
        then normalized to ``uint8`` using :func:`cv2.normalize`.
    """
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not normalize:
        return img
    if scale_minmax is not None:
        lo, hi = scale_minmax
        img = img.astype(np.float32)
        img = np.clip(img, lo, hi)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

def imread_color(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read {path}")
    return img

def file_times_minutes(paths: list[Path]) -> list[float]:
    # Compute minutes elapsed from first frame using file modification timestamps
    if not paths: return []
    mtimes = [p.stat().st_mtime for p in paths]
    t0 = mtimes[0]
    return [ (t - t0)/60.0 for t in mtimes ]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
