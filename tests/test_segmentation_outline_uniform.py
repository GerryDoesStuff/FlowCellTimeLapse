import numpy as np
import sys
from pathlib import Path

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import app.core.segmentation as segmod


def _uniform_bh(gray, invert=True):
    return np.full_like(gray, 5, dtype=np.uint8)


def test_adaptive_skips_near_uniform_outline(monkeypatch):
    img = np.array(
        [[10, 60, 200],
         [100, 150, 250],
         [30, 180, 220]],
        dtype=np.uint8,
    )
    monkeypatch.setattr(segmod, "outline_focused", _uniform_bh)
    seg = segmod.segment(
        img,
        method="adaptive",
        invert=False,
        morph_open_radius=0,
        morph_close_radius=0,
    )
    seg_skip = segmod.segment(
        img,
        method="adaptive",
        invert=False,
        skip_outline=True,
        morph_open_radius=0,
        morph_close_radius=0,
    )
    assert np.array_equal(seg, seg_skip)


def test_local_skips_near_uniform_outline(monkeypatch):
    img = np.array(
        [[10, 60, 200],
         [100, 150, 250],
         [30, 180, 220]],
        dtype=np.uint8,
    )
    monkeypatch.setattr(segmod, "outline_focused", _uniform_bh)
    seg = segmod.segment(
        img,
        method="local",
        invert=False,
        morph_open_radius=0,
        morph_close_radius=0,
    )
    seg_skip = segmod.segment(
        img,
        method="local",
        invert=False,
        skip_outline=True,
        morph_open_radius=0,
        morph_close_radius=0,
    )
    assert np.array_equal(seg, seg_skip)
