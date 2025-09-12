import numpy as np
import sys
from pathlib import Path
from skimage import filters

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment


def test_multi_otsu_segmentation_two_classes():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[:, 2:] = 200

    seg = segment(
        img,
        method="multi_otsu",
        invert=False,
        skip_outline=True,
        morph_open_radius=0,
        morph_close_radius=0,
        remove_objects_smaller_px=0,
        remove_holes_smaller_px=0,
    )

    t = filters.threshold_multiotsu(img, classes=2)
    expected = (img >= t[0]).astype(np.uint8)

    assert np.array_equal(seg, expected)


def test_multi_otsu_uniform_image_returns_empty_mask():
    """Uniform images should not raise and should produce an empty mask."""
    img = np.full((5, 5), 128, dtype=np.uint8)

    seg = segment(
        img,
        method="multi_otsu",
        invert=False,
        skip_outline=True,
        morph_open_radius=0,
        morph_close_radius=0,
        remove_objects_smaller_px=0,
        remove_holes_smaller_px=0,
    )

    assert np.count_nonzero(seg) == 0
