from __future__ import annotations
import numpy as np
import cv2
from typing import List

def estimate_temporal_background(frames: list[np.ndarray], n_early: int=5) -> np.ndarray:
    # Use median of the earliest n frames as background proxy
    n = min(n_early, len(frames))
    stack = np.stack(frames[:n], axis=0).astype(np.float32)
    med = np.median(stack, axis=0).astype(np.uint8)
    return med

def normalize_background(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    # Simple illumination correction: subtract, then rescale
    diff = cv2.subtract(img, bg)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return diff
