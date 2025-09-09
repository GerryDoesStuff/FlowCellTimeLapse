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

def imread_gray(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

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
