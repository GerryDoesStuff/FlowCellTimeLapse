import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.evaluation import evaluate_diff_masks


def create_masks(tmp_path: Path) -> Path:
    diff_dir = tmp_path / "diff"
    (diff_dir / "bw").mkdir(parents=True)
    (diff_dir / "new").mkdir(parents=True)
    (diff_dir / "lost").mkdir(parents=True)

    new_mask = np.zeros((4, 4), dtype=np.uint8)
    new_mask[0, 0] = 255
    new_mask[1, 1] = 255

    lost_mask = np.zeros((4, 4), dtype=np.uint8)
    lost_mask[0, 1] = 255

    diff_mask = np.zeros((4, 4), dtype=np.uint8)
    diff_mask[0, 0] = 255
    diff_mask[1, 1] = 255
    diff_mask[0, 1] = 255

    cv2.imwrite(str(diff_dir / "new" / "0000_bw_new.png"), new_mask)
    cv2.imwrite(str(diff_dir / "lost" / "0000_bw_lost.png"), lost_mask)
    cv2.imwrite(str(diff_dir / "bw" / "0001_bw_diff.png"), diff_mask)
    return diff_dir


def test_evaluate_diff_masks(tmp_path):
    diff_dir = create_masks(tmp_path)
    df = evaluate_diff_masks(diff_dir, csv_path="summary.csv")
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert row["area_new_px"] == 2
    assert row["area_lost_px"] == 1
    assert row["net_new_px"] == 1
    assert row["area_diff_px"] == 3
    assert (diff_dir / "summary.csv").exists()
