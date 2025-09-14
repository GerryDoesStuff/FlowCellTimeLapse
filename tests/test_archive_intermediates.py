import os
import sys
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import zipfile

# Ensure modules are importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.workers.pipeline_worker import PipelineWorker


def test_archive_intermediate_dirs(monkeypatch, tmp_path):
    # Create dummy images so worker has paths
    img = np.zeros((5, 5), dtype=np.uint8)
    paths = []
    for i in range(2):
        p = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    # Stub analyze_sequence to write out intermediate dirs
    def fake_analyze(paths, reg_cfg, seg_cfg, app_cfg, out_dir):
        for name in ["registered", "diff", "overlay"]:
            d = out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "dummy.png"), img)
        return pd.DataFrame()

    monkeypatch.setattr("app.workers.pipeline_worker.analyze_sequence", fake_analyze)

    reg_cfg = {}
    seg_cfg = {}
    app_cfg = {"direction": "first-to-last", "save_intermediates": True, "archive_intermediates": True}
    out_dir = tmp_path / "out"
    worker = PipelineWorker(paths, reg_cfg, seg_cfg, app_cfg, out_dir)
    worker.run()

    for name in ["registered", "diff", "overlay"]:
        assert not (out_dir / name).exists()
        zpath = out_dir / f"{name}.zip"
        assert zpath.exists()
        with zipfile.ZipFile(zpath) as zf:
            assert any(member.endswith("dummy.png") for member in zf.namelist())
