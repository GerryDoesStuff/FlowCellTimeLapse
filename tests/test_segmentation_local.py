import numpy as np
import sys
from pathlib import Path
from skimage import filters
import cv2

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment, outline_focused


def test_local_segmentation_matches_skimage_threshold_local():
    img = np.array(
        [[10, 60, 200],
         [100, 150, 250],
         [30, 180, 220]],
        dtype=np.uint8,
    )

    seg = segment(
        img,
        method="local",
        invert=False,
        local_block=3,
        morph_open_radius=0,
        morph_close_radius=0,
        remove_objects_smaller_px=0,
        remove_holes_smaller_px=0,
    )

    feat = outline_focused(img, invert=False)
    loc = filters.threshold_local(feat, 3)
    expected = (feat > loc).astype(np.uint8)

    assert np.array_equal(seg, expected)


def test_local_segmentation_use_diff_matches_skimage_threshold_local():
    img = np.array(
        [[10, 60, 200],
         [100, 150, 250],
         [30, 180, 220]],
        dtype=np.uint8,
    )

    seg = segment(
        img,
        method="local",
        invert=False,
        use_diff=True,
        local_block=3,
        morph_open_radius=0,
        morph_close_radius=0,
        remove_objects_smaller_px=0,
        remove_holes_smaller_px=0,
    )

    feat = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    loc = filters.threshold_local(feat, 3)
    expected = (feat > loc).astype(np.uint8)

    assert np.array_equal(seg, expected)
