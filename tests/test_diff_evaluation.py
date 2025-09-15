import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

import app.core.evaluation as evaluation
from app.core.evaluation import evaluate_diff_masks


def create_masks(tmp_path: Path) -> Path:
    diff_dir = tmp_path / "diff"
    (diff_dir / "bw").mkdir(parents=True)
    (diff_dir / "new").mkdir(parents=True)
    (diff_dir / "lost").mkdir(parents=True)
    (diff_dir / "gain").mkdir(parents=True)
    (diff_dir / "loss").mkdir(parents=True)

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
    cv2.imwrite(str(diff_dir / "gain" / "0000_bw_gain.png"), cv2.bitwise_and(new_mask, diff_mask))
    cv2.imwrite(str(diff_dir / "loss" / "0000_bw_loss.png"), cv2.bitwise_and(lost_mask, diff_mask))
    return diff_dir


def test_diff_evaluation(tmp_path, monkeypatch):
    diff_dir = create_masks(tmp_path)

    accessed_dirs: list[Path] = []
    read_files: list[Path] = []

    original_glob = Path.glob

    def tracking_glob(self, pattern):
        accessed_dirs.append(self)
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", tracking_glob)

    original_read = evaluation._read_mask

    def tracking_read(path):
        if path is not None:
            read_files.append(Path(path))
        return original_read(path)

    monkeypatch.setattr(evaluation, "_read_mask", tracking_read)

    df = evaluate_diff_masks(diff_dir)

    assert df.shape[0] == 1
    row = df.iloc[0]
    assert row["area_new_px"] == 2
    assert row["area_lost_px"] == 1
    assert row["net_new_px"] == 1
    assert row["area_diff_px"] == 3

    diff_dir_res = diff_dir.resolve()
    all_paths = [p.resolve() for p in accessed_dirs + read_files]
    assert all(p.is_relative_to(diff_dir_res) for p in all_paths)
