from __future__ import annotations
from PyQt6.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
                             QPushButton, QHBoxLayout, QLabel, QCheckBox, QComboBox, QSpinBox,
                             QDoubleSpinBox, QSlider, QGroupBox, QFormLayout, QLineEdit)
from PyQt6.QtCore import Qt, QThread
from pathlib import Path
import json
import pyqtgraph as pg
import numpy as np
import pandas as pd
import cv2

from ..models.config import RegParams, SegParams, AppParams, save_settings, load_settings, save_preset, load_preset
from ..core.io_utils import discover_images, imread_gray, file_times_minutes
from ..core.registration import register_ecc, register_orb
from ..core.segmentation import segment
from ..core.processing import overlay_outline
from ..workers.pipeline_worker import PipelineWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yeast Flowcell Analyzer — PyQt")
        self.resize(1200, 800)
        self.reg, self.seg, self.app = load_settings()
        self.paths: list[Path] = []
        self._build_ui()
        # Cached preview images for alpha blending
        self._reg_ref = None
        self._reg_warp = None
        self._seg_gray = None
        self._seg_overlay = None
        self._current_preview = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: controls
        controls = QVBoxLayout()
        layout.addLayout(controls, 0)

        # Folder
        folder_box = QHBoxLayout()
        self.folder_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._choose_folder)
        folder_box.addWidget(self.folder_edit)
        folder_box.addWidget(browse_btn)
        controls.addLayout(folder_box)

        # Reference + timestamps
        ref_group = QGroupBox("Reference & Timing")
        ref_form = QFormLayout(ref_group)
        self.ref_combo = QComboBox(); self.ref_combo.addItems(["last","first","middle","custom"])
        self.ref_combo.setCurrentText(self.app.reference_choice)
        self.ref_idx = QSpinBox(); self.ref_idx.setMinimum(0); self.ref_idx.setValue(self.app.custom_ref_index)
        self.use_ts = QCheckBox("Use file timestamps for frame spacing"); self.use_ts.setChecked(self.app.use_file_timestamps)
        self.dt_min = QDoubleSpinBox(); self.dt_min.setDecimals(3); self.dt_min.setMinimum(0.0); self.dt_min.setValue(self.app.minutes_between_frames)
        ref_form.addRow("Reference frame", self.ref_combo)
        ref_form.addRow("Custom index (0-based)", self.ref_idx)
        ref_form.addRow(self.use_ts)
        ref_form.addRow("Fallback Δt (min)", self.dt_min)
        controls.addWidget(ref_group)

        # Registration params
        reg_group = QGroupBox("Registration")
        reg_form = QFormLayout(reg_group)
        self.reg_method = QComboBox(); self.reg_method.addItems(["ECC","ORB"]); self.reg_method.setCurrentText(self.reg.method)
        self.reg_model = QComboBox(); self.reg_model.addItems(["translation","euclidean","affine","homography"]); self.reg_model.setCurrentText(self.reg.model)
        self.max_iters = QSpinBox(); self.max_iters.setRange(10, 10000); self.max_iters.setValue(self.reg.max_iters)
        self.eps = QDoubleSpinBox(); self.eps.setDecimals(9); self.eps.setSingleStep(1e-6); self.eps.setValue(self.reg.eps)
        self.use_masked = QCheckBox("Use masked ECC"); self.use_masked.setChecked(self.reg.use_masked_ecc)
        reg_form.addRow("Method", self.reg_method)
        reg_form.addRow("Model", self.reg_model)
        reg_form.addRow("Max iters", self.max_iters)
        reg_form.addRow("Epsilon", self.eps)
        reg_form.addRow(self.use_masked)
        controls.addWidget(reg_group)

        # Segmentation params
        seg_group = QGroupBox("Segmentation")
        seg_form = QFormLayout(seg_group)
        self.seg_method = QComboBox(); self.seg_method.addItems(["otsu","adaptive","local","manual"]); self.seg_method.setCurrentText(self.seg.method)
        self.invert = QCheckBox("Cells darker (invert)"); self.invert.setChecked(self.seg.invert)
        self.manual_t = QSpinBox(); self.manual_t.setRange(0,255); self.manual_t.setValue(self.seg.manual_thresh)
        self.adaptive_blk = QSpinBox(); self.adaptive_blk.setRange(3,999); self.adaptive_blk.setSingleStep(2); self.adaptive_blk.setValue(self.seg.adaptive_block)
        self.adaptive_C = QSpinBox(); self.adaptive_C.setRange(-100,100); self.adaptive_C.setValue(self.seg.adaptive_C)
        self.local_blk = QSpinBox(); self.local_blk.setRange(3,999); self.local_blk.setSingleStep(2); self.local_blk.setValue(self.seg.local_block)
        self.open_r = QSpinBox(); self.open_r.setRange(0,50); self.open_r.setValue(self.seg.morph_open_radius)
        self.close_r = QSpinBox(); self.close_r.setRange(0,50); self.close_r.setValue(self.seg.morph_close_radius)
        self.rm_obj = QSpinBox(); self.rm_obj.setRange(0,100000); self.rm_obj.setValue(self.seg.remove_objects_smaller_px)
        self.rm_holes = QSpinBox(); self.rm_holes.setRange(0,100000); self.rm_holes.setValue(self.seg.remove_holes_smaller_px)
        seg_form.addRow("Method", self.seg_method)
        seg_form.addRow(self.invert)
        seg_form.addRow("Manual threshold", self.manual_t)
        seg_form.addRow("Adaptive block", self.adaptive_blk)
        seg_form.addRow("Adaptive C", self.adaptive_C)
        seg_form.addRow("Local block", self.local_blk)
        seg_form.addRow("Open radius", self.open_r)
        seg_form.addRow("Close radius", self.close_r)
        seg_form.addRow("Remove objects < px", self.rm_obj)
        seg_form.addRow("Remove holes < px", self.rm_holes)
        seg_preview_btn = QPushButton("Preview Segmentation")
        seg_preview_btn.clicked.connect(self._preview_segmentation)
        seg_form.addRow(seg_preview_btn)
        controls.addWidget(seg_group)

        # Presets
        preset_box = QHBoxLayout()
        save_p = QPushButton("Save Preset"); load_p = QPushButton("Load Preset")
        save_p.clicked.connect(self._save_preset); load_p.clicked.connect(self._load_preset)
        preset_box.addWidget(save_p); preset_box.addWidget(load_p)
        controls.addLayout(preset_box)

        # Run / Preview
        run_box = QHBoxLayout()
        run_btn = QPushButton("Run Analysis")
        run_btn.clicked.connect(self._run_pipeline)
        preview_btn = QPushButton("Preview Registration")
        preview_btn.clicked.connect(self._preview_registration)
        run_box.addWidget(run_btn)
        run_box.addWidget(preview_btn)
        controls.addLayout(run_box)

        controls.addStretch(1)

        # Right: viewer
        right = QVBoxLayout()
        layout.addLayout(right, 1)

        self.view = pg.ImageView()
        right.addWidget(self.view)

        overlay_box = QHBoxLayout()
        self.overlay_ref_cb = QCheckBox("Show reference overlay"); self.overlay_ref_cb.setChecked(True)
        self.overlay_mov_cb = QCheckBox("Show moving overlay"); self.overlay_mov_cb.setChecked(True)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal); self.alpha_slider.setRange(0,100); self.alpha_slider.setValue(50)
        overlay_box.addWidget(self.overlay_ref_cb)
        overlay_box.addWidget(self.overlay_mov_cb)
        overlay_box.addWidget(QLabel("Opacity"))
        overlay_box.addWidget(self.alpha_slider)
        right.addLayout(overlay_box)

        # Refresh overlays when controls change
        self.alpha_slider.valueChanged.connect(self._refresh_overlay_alpha)
        self.overlay_ref_cb.toggled.connect(self._refresh_overlay_alpha)
        self.overlay_mov_cb.toggled.connect(self._refresh_overlay_alpha)

        # Status
        self.status_label = QLabel("Ready.")
        right.addWidget(self.status_label)

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select image folder", "")
        if d:
            self.folder_edit.setText(d)
            self.paths = discover_images(Path(d))
            if not self.paths:
                QMessageBox.warning(self, "No images", "No images found.")
                return
            # show first for preview
            img = imread_gray(self.paths[0])
            self.view.setImage(img.T)  # pyqtgraph expects column-major by default
            self.status_label.setText(f"Found {len(self.paths)} images. First: {self.paths[0].name}")

    def _collect_params(self):
        # Pull from UI into dataclasses
        reg = RegParams(method=self.reg_method.currentText(),
                        model=self.reg_model.currentText(),
                        max_iters=self.max_iters.value(),
                        eps=self.eps.value(),
                        use_masked_ecc=self.use_masked.isChecked())
        seg = SegParams(method=self.seg_method.currentText(),
                        invert=self.invert.isChecked(),
                        manual_thresh=self.manual_t.value(),
                        adaptive_block=self.adaptive_blk.value(),
                        adaptive_C=self.adaptive_C.value(),
                        local_block=self.local_blk.value(),
                        morph_open_radius=self.open_r.value(),
                        morph_close_radius=self.close_r.value(),
                        remove_objects_smaller_px=self.rm_obj.value(),
                        remove_holes_smaller_px=self.rm_holes.value())
        app = AppParams(reference_choice=self.ref_combo.currentText(),
                        custom_ref_index=self.ref_idx.value(),
                        minutes_between_frames=self.dt_min.value(),
                        use_file_timestamps=self.use_ts.isChecked())
        save_settings(reg, seg, app)
        return reg, seg, app

    def _save_preset(self):
        reg, seg, app = self._collect_params()
        path, _ = QFileDialog.getSaveFileName(self, "Save Preset", "preset.json", "JSON (*.json)")
        if path:
            save_preset(path, reg, seg, app)
            self.status_label.setText(f"Preset saved: {path}")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON (*.json)")
        if not path: return
        reg, seg, app = load_preset(path)
        self.reg = reg; self.seg = seg; self.app = app
        # Update UI
        self.reg_method.setCurrentText(reg.method)
        self.reg_model.setCurrentText(reg.model)
        self.max_iters.setValue(reg.max_iters)
        self.eps.setValue(reg.eps)
        self.use_masked.setChecked(reg.use_masked_ecc)
        self.seg_method.setCurrentText(seg.method)
        self.invert.setChecked(seg.invert)
        self.manual_t.setValue(seg.manual_thresh)
        self.adaptive_blk.setValue(seg.adaptive_block)
        self.adaptive_C.setValue(seg.adaptive_C)
        self.local_blk.setValue(seg.local_block)
        self.open_r.setValue(seg.morph_open_radius)
        self.close_r.setValue(seg.morph_close_radius)
        self.rm_obj.setValue(seg.remove_objects_smaller_px)
        self.rm_holes.setValue(seg.remove_holes_smaller_px)
        self.ref_combo.setCurrentText(app.reference_choice)
        self.ref_idx.setValue(app.custom_ref_index)
        self.dt_min.setValue(app.minutes_between_frames)
        self.use_ts.setChecked(app.use_file_timestamps)
        self.status_label.setText(f"Preset loaded: {path}")

    def _refresh_overlay_alpha(self):
        """Blend cached overlays according to slider and checkbox states."""
        alpha = self.alpha_slider.value() / 100.0
        if self._current_preview == "registration" and self._reg_ref is not None and self._reg_warp is not None:
            if self.overlay_ref_cb.isChecked() and self.overlay_mov_cb.isChecked():
                blend = cv2.addWeighted(self._reg_ref, 1 - alpha, self._reg_warp, alpha, 0)
            elif self.overlay_ref_cb.isChecked():
                blend = self._reg_ref
            elif self.overlay_mov_cb.isChecked():
                blend = self._reg_warp
            else:
                blend = np.zeros_like(self._reg_ref)
            color = cv2.cvtColor(blend, cv2.COLOR_GRAY2BGR)
            self.view.setImage(color.transpose(1, 0, 2))
        elif self._current_preview == "segmentation" and self._seg_gray is not None and self._seg_overlay is not None:
            blend = cv2.addWeighted(self._seg_gray, 1 - alpha, self._seg_overlay, alpha, 0)
            self.view.setImage(blend.transpose(1, 0, 2))

    def _preview_registration(self):
        if len(self.paths) < 2:
            QMessageBox.warning(self, "Need at least two images", "Load at least two images for preview.")
            return
        try:
            reg, _, app = self._collect_params()
            if app.reference_choice == "last":
                ref_idx = len(self.paths) - 1
            elif app.reference_choice == "first":
                ref_idx = 0
            elif app.reference_choice == "middle":
                ref_idx = len(self.paths) // 2
            else:
                ref_idx = app.custom_ref_index
            ref_img = imread_gray(self.paths[ref_idx])
            mov_idx = 0 if ref_idx != 0 else 1
            mov_img = imread_gray(self.paths[mov_idx])
            if reg.method.upper() == "ORB":
                _, warped, _ = register_orb(ref_img, mov_img, model=reg.model)
            else:
                _, warped, _ = register_ecc(ref_img, mov_img, model=reg.model,
                                            max_iters=reg.max_iters, eps=reg.eps)
            self._reg_ref = ref_img
            self._reg_warp = warped
            self._current_preview = "registration"
            self._refresh_overlay_alpha()
            self.status_label.setText("Preview successful.")
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")

    def _preview_segmentation(self):
        if not self.paths:
            QMessageBox.warning(self, "No images", "Choose an image folder first.")
            return
        try:
            _, seg, app = self._collect_params()
            if app.reference_choice == "last":
                idx = len(self.paths) - 1
            elif app.reference_choice == "first":
                idx = 0
            elif app.reference_choice == "middle":
                idx = len(self.paths) // 2
            else:
                idx = app.custom_ref_index
            gray = imread_gray(self.paths[idx])
            bw = segment(gray,
                         method=seg.method,
                         invert=seg.invert,
                         manual_thresh=seg.manual_thresh,
                         adaptive_block=seg.adaptive_block,
                         adaptive_C=seg.adaptive_C,
                         local_block=seg.local_block,
                         morph_open_radius=seg.morph_open_radius,
                         morph_close_radius=seg.morph_close_radius,
                         remove_objects_smaller_px=seg.remove_objects_smaller_px,
                         remove_holes_smaller_px=seg.remove_holes_smaller_px)
            self._seg_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self._seg_overlay = overlay_outline(gray, bw)
            self._current_preview = "segmentation"
            self._refresh_overlay_alpha()
            self.status_label.setText("Segmentation preview successful.")
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")

    def _run_pipeline(self):
        if not self.paths:
            QMessageBox.warning(self, "No images", "Choose an image folder first.")
            return

        reg, seg, app = self._collect_params()
        # Build slim dicts for worker
        reg_cfg = dict(method=reg.method, model=reg.model, max_iters=reg.max_iters, eps=reg.eps, use_masked_ecc=reg.use_masked_ecc)
        seg_cfg = dict(method=seg.method, invert=seg.invert, manual_thresh=seg.manual_thresh,
                       adaptive_block=seg.adaptive_block, adaptive_C=seg.adaptive_C, local_block=seg.local_block,
                       morph_open_radius=seg.morph_open_radius, morph_close_radius=seg.morph_close_radius,
                       remove_objects_smaller_px=seg.remove_objects_smaller_px, remove_holes_smaller_px=seg.remove_holes_smaller_px)
        app_cfg = dict(reference_choice=app.reference_choice, custom_ref_index=app.custom_ref_index,
                       use_difference_for_seg=False, save_intermediates=True)

        # timestamps if requested
        if app.use_file_timestamps:
            minutes = file_times_minutes(self.paths)
            # (currently minutes not directly used in processing; kept for CSV in future patch)

        out_dir = Path(self.folder_edit.text()) / "_processed_pyqt"
        self.thread = QThread()
        self.worker = PipelineWorker(self.paths, reg_cfg, seg_cfg, app_cfg, out_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.thread.start()
        self.status_label.setText("Processing…")

    def _on_done(self, out_dir: str):
        self.status_label.setText(f"Done. Outputs: {out_dir}")
        self.thread.quit(); self.thread.wait()

    def _on_failed(self, err: str):
        QMessageBox.critical(self, "Error", err)
        self.status_label.setText("Failed.")
        self.thread.quit(); self.thread.wait()
