import numpy as np
import sys
from pathlib import Path
import cv2

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment, outline_focused


def test_otsu_segmentation_matches_cv2_threshold():
    img = np.array(
        [[10, 60, 200],
         [100, 150, 250],
         [30, 180, 220]],
        dtype=np.uint8,
    )

    seg = segment(
        img,
        method="otsu",
        invert=False,
        morph_open_radius=0,
        morph_close_radius=0,
        remove_objects_smaller_px=0,
        remove_holes_smaller_px=0,
    )

    feat = outline_focused(img, invert=False)
    _, th = cv2.threshold(feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    expected = (th > 0).astype(np.uint8)

    assert np.array_equal(seg, expected)

