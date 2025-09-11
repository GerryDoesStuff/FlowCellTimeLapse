import numpy as np
import sys
from pathlib import Path
import cv2

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment, outline_focused


def test_adaptive_segmentation_matches_cv2_adaptive_threshold():
    img = np.array(
        [[10, 60, 200],
         [100, 150, 250],
         [30, 180, 220]],
        dtype=np.uint8,
    )

    seg = segment(
        img,
        method="adaptive",
        invert=False,
        adaptive_block=3,
        adaptive_C=5,
        morph_open_radius=0,
        morph_close_radius=0,
        remove_objects_smaller_px=0,
        remove_holes_smaller_px=0,
    )

    feat = outline_focused(img, invert=False)
    th = cv2.adaptiveThreshold(
        feat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        3,
        5,
    )
    expected = (th > 0).astype(np.uint8)

    assert np.array_equal(seg, expected)
