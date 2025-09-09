from __future__ import annotations
from PyQt6.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
                             QPushButton, QHBoxLayout, QLabel, QCheckBox, QComboBox, QSpinBox,
                             QDoubleSpinBox, QSlider, QGroupBox, QFormLayout, QLineEdit, QToolTip,
                             QColorDialog)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QThread, QTimer
from pathlib import Path
import json
import pyqtgraph as pg
import numpy as np
import pandas as pd
import cv2

from ..models.config import RegParams, SegParams, AppParams, save_settings, load_settings, save_preset, load_preset
from ..core.io_utils import discover_images, imread_gray, file_times_minutes, compute_global_minmax
from ..core.registration import register_ecc, register_orb, preprocess
from ..core.segmentation import segment
from ..core.processing import overlay_outline
from ..workers.pipeline_worker import PipelineWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yeast Flowcell Analyzer — PyQt")
        self.resize(1200, 800)
        self.reg, self.seg, self.app = load_settings()
        self.ref_color = tuple(self.app.overlay_ref_color)
        self.mov_color = tuple(self.app.overlay_mov_color)
        self.paths: list[Path] = []
        # Cached preview images for alpha blending
        self._reg_ref = None
        self._reg_warp = None
        self._seg_gray = None
        self._seg_overlay = None
        self._current_preview = None
        self._build_ui()
        self._param_timer = QTimer(self)
        self._param_timer.setSingleShot(True)
        self._param_timer.setInterval(200)
        self._param_timer.timeout.connect(self._apply_param_change)
        # Track whether registration has been run so segmentation preview can
        # be gated until then.
        self._registration_done = False

    def _add_help(self, widget, text: str) -> None:
        widget.setToolTip(text)
        widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        widget.customContextMenuRequested.connect(
            lambda pos, w=widget, t=text: QToolTip.showText(w.mapToGlobal(pos), t, w)
        )

    def _set_btn_color(self, btn: QPushButton, color: tuple[int, int, int]) -> None:
        btn.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]});")

    def _choose_color(self, which: str) -> None:
        initial = self.ref_color if which == 'ref' else self.mov_color
        col = QColorDialog.getColor(QColor(*initial), self, "Select color")
        if col.isValid():
            color = (col.red(), col.green(), col.blue())
            if which == 'ref':
                self.ref_color = color
                self._set_btn_color(self.ref_color_btn, color)
            else:
                self.mov_color = color
                self._set_btn_color(self.mov_color_btn, color)
            self._refresh_overlay_alpha()
            self._persist_settings()

    def _on_overlay_mode_changed(self, mode: str) -> None:
        custom = mode == 'custom'
        for w in (self.ref_color_btn, self.mov_color_btn, self.ref_color_label, self.mov_color_label):
            w.setVisible(custom)
        self._refresh_overlay_alpha()

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
        if self.app.last_folder:
            self.folder_edit.setText(self.app.last_folder)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._choose_folder)
        folder_box.addWidget(self.folder_edit)
        folder_box.addWidget(browse_btn)
        controls.addLayout(folder_box)
        self.folder_edit.textChanged.connect(self._persist_settings)

        # Direction + timestamps
        ref_group = QGroupBox("Direction & Timing")
        ref_form = QFormLayout(ref_group)
        self.dir_combo = QComboBox(); self.dir_combo.addItems(["last-to-first", "first-to-last"])
        self.dir_combo.setCurrentText(self.app.direction)
        self.use_ts = QCheckBox("Use file timestamps for frame spacing"); self.use_ts.setChecked(self.app.use_file_timestamps)
        self.dt_min = QDoubleSpinBox(); self.dt_min.setDecimals(3); self.dt_min.setMinimum(0.0); self.dt_min.setValue(self.app.minutes_between_frames)
        ref_form.addRow("Analysis direction", self.dir_combo)
        ref_form.addRow(self.use_ts)
        ref_form.addRow("Fallback Δt (min)", self.dt_min)
        controls.addWidget(ref_group)
        self.dir_combo.currentTextChanged.connect(self._show_reference_frame)
        self.dir_combo.currentTextChanged.connect(self._persist_settings)
        self.use_ts.toggled.connect(self._persist_settings)
        self.dt_min.valueChanged.connect(self._persist_settings)

        # Intensity scaling
        intensity_group = QGroupBox("Intensity")
        intensity_form = QFormLayout(intensity_group)
        self.norm_cb = QCheckBox("Normalize frames")
        self.norm_cb.setChecked(self.app.normalize)
        self.scale_min = QSpinBox(); self.scale_min.setRange(-1000000, 1000000)
        self.scale_max = QSpinBox(); self.scale_max.setRange(-1000000, 1000000)
        if self.app.scale_minmax is not None:
            self.scale_min.setValue(int(self.app.scale_minmax[0]))
            self.scale_max.setValue(int(self.app.scale_minmax[1]))
        else:
            self.scale_min.setValue(0)
            self.scale_max.setValue(0)
        self.rescale_cb = QCheckBox("Rescale background subtraction")
        self.rescale_cb.setChecked(self.app.rescale_background)
        intensity_form.addRow(self.norm_cb)
        intensity_form.addRow("Min", self.scale_min)
        intensity_form.addRow("Max", self.scale_max)
        self.auto_scale_btn = QPushButton("Auto min/max")
        self.auto_scale_btn.clicked.connect(self._auto_scale_minmax)
        intensity_form.addRow(self.auto_scale_btn)
        intensity_form.addRow(self.rescale_cb)
        controls.addWidget(intensity_group)
        self.scale_min.setEnabled(self.norm_cb.isChecked())
        self.scale_max.setEnabled(self.norm_cb.isChecked())
        self.norm_cb.toggled.connect(lambda v: [self.scale_min.setEnabled(v), self.scale_max.setEnabled(v)])
        self.norm_cb.toggled.connect(self._persist_settings)
        self.scale_min.valueChanged.connect(self._persist_settings)
        self.scale_max.valueChanged.connect(self._persist_settings)
        self.rescale_cb.toggled.connect(self._persist_settings)
        self.norm_cb.toggled.connect(self._on_params_changed)
        self.scale_min.valueChanged.connect(self._on_params_changed)
        self.scale_max.valueChanged.connect(self._on_params_changed)
        self.rescale_cb.toggled.connect(self._on_params_changed)

        # Registration params
        reg_group = QGroupBox("Registration")
        reg_form = QFormLayout(reg_group)
        self.reg_method = QComboBox(); self.reg_method.addItems(["ECC","ORB"]); self.reg_method.setCurrentText(self.reg.method)
        self.reg_model = QComboBox(); self.reg_model.addItems(["translation","euclidean","affine","homography"]); self.reg_model.setCurrentText(self.reg.model)
        self.max_iters = QSpinBox(); self.max_iters.setRange(10, 10000); self.max_iters.setValue(self.reg.max_iters)
        self.eps = QDoubleSpinBox(); self.eps.setDecimals(9); self.eps.setSingleStep(1e-6); self.eps.setValue(self.reg.eps)
        self.gauss_sigma = QDoubleSpinBox(); self.gauss_sigma.setRange(0.0, 10.0); self.gauss_sigma.setDecimals(2); self.gauss_sigma.setSingleStep(0.1); self.gauss_sigma.setValue(self.reg.gauss_blur_sigma)
        self.clahe_clip = QDoubleSpinBox(); self.clahe_clip.setRange(0.0, 40.0); self.clahe_clip.setDecimals(2); self.clahe_clip.setSingleStep(0.1); self.clahe_clip.setValue(self.reg.clahe_clip)
        self.clahe_grid = QSpinBox(); self.clahe_grid.setRange(1, 64); self.clahe_grid.setValue(self.reg.clahe_grid)
        self.init_radius = QSpinBox(); self.init_radius.setRange(1, 1000); self.init_radius.setValue(self.reg.initial_radius)
        self.growth_factor = QDoubleSpinBox(); self.growth_factor.setRange(0.1, 10.0); self.growth_factor.setDecimals(2); self.growth_factor.setSingleStep(0.1); self.growth_factor.setValue(self.reg.growth_factor)
        self.use_masked = QCheckBox("Use masked ECC"); self.use_masked.setChecked(self.reg.use_masked_ecc)
        self.orb_features = QSpinBox(); self.orb_features.setRange(1, 100000); self.orb_features.setValue(self.reg.orb_features)
        self.match_ratio = QDoubleSpinBox(); self.match_ratio.setRange(0.0, 1.0); self.match_ratio.setDecimals(2); self.match_ratio.setSingleStep(0.05); self.match_ratio.setValue(self.reg.match_ratio)
        reg_form.addRow("Method", self.reg_method)
        reg_form.addRow("Model", self.reg_model)
        reg_form.addRow("Max iters", self.max_iters)
        self.max_iters_label = reg_form.labelForField(self.max_iters)
        reg_form.addRow("Epsilon", self.eps)
        self.eps_label = reg_form.labelForField(self.eps)
        reg_form.addRow("Gaussian σ", self.gauss_sigma)
        reg_form.addRow("CLAHE clip", self.clahe_clip)
        reg_form.addRow("CLAHE grid", self.clahe_grid)
        reg_form.addRow("Initial radius", self.init_radius)
        reg_form.addRow("Growth factor", self.growth_factor)
        reg_form.addRow("ORB features", self.orb_features)
        self.orb_features_label = reg_form.labelForField(self.orb_features)
        reg_form.addRow("Match ratio", self.match_ratio)
        self.match_ratio_label = reg_form.labelForField(self.match_ratio)
        reg_form.addRow(self.use_masked)
        self._add_help(
            self.reg_method,
            "Registration algorithm. ECC correlates intensities for higher accuracy but "
            "is slower; ORB matches keypoints for speed and robustness to large "
            "motions."
        )
        self._add_help(
            self.reg_model,
            "Geometric transform model. Translation is fastest; Euclidean adds rotation; "
            "Affine adds shear/scale; Homography handles perspective but is slowest."
        )
        self._add_help(
            self.max_iters,
            "Maximum ECC iterations. More iterations improve alignment but slow "
            "processing. Typical range: 50–300."
        )
        self._add_help(
            self.eps,
            "ECC convergence threshold. Smaller values yield more precise results at "
            "the cost of extra iterations. Recommended: 1e-4–1e-6."
        )
        self._add_help(
            self.gauss_sigma,
            "Gaussian blur σ before registration to reduce noise. 0–2 is common; "
            "higher values smooth detail (faster, less accurate)."
        )
        self._add_help(
            self.clahe_clip,
            "CLAHE clip limit for local contrast enhancement. 0 disables. Typical "
            "range: 0–5; higher improves contrast but may amplify noise."
        )
        self._add_help(
            self.clahe_grid,
            "CLAHE tile grid size. Smaller (8–16) boosts local detail but may add "
            "artifacts; larger (up to 32) is smoother but less adaptive."
        )
        self._add_help(self.init_radius, "Initial search window radius in pixels for ECC.")
        self._add_help(
            self.growth_factor,
            "Scale search window after each registration step (>=1 keeps more context)."
        )
        self._add_help(self.orb_features, "Number of ORB features to detect.")
        self._add_help(self.match_ratio, "Lowe ratio for filtering ORB matches.")
        self._add_help(
            self.use_masked,
            "Use segmentation mask during ECC to focus on cells, improving "
            "accuracy in cluttered scenes but requiring prior segmentation."
        )
        reg_preview_btn = QPushButton("Preview Registration")
        reg_preview_btn.clicked.connect(self._preview_registration)
        reg_form.addRow(reg_preview_btn)
        controls.addWidget(reg_group)
        self.reg_method.currentTextChanged.connect(self._persist_settings)
        self.reg_model.currentTextChanged.connect(self._persist_settings)
        self.max_iters.valueChanged.connect(self._persist_settings)
        self.eps.valueChanged.connect(self._persist_settings)
        self.gauss_sigma.valueChanged.connect(self._persist_settings)
        self.clahe_clip.valueChanged.connect(self._persist_settings)
        self.clahe_grid.valueChanged.connect(self._persist_settings)
        self.init_radius.valueChanged.connect(self._persist_settings)
        self.growth_factor.valueChanged.connect(self._persist_settings)
        self.orb_features.valueChanged.connect(self._persist_settings)
        self.match_ratio.valueChanged.connect(self._persist_settings)
        self.use_masked.toggled.connect(self._persist_settings)
        self.reg_method.currentTextChanged.connect(self._on_params_changed)
        self.reg_model.currentTextChanged.connect(self._on_params_changed)
        self.max_iters.valueChanged.connect(self._on_params_changed)
        self.eps.valueChanged.connect(self._on_params_changed)
        self.gauss_sigma.valueChanged.connect(self._on_params_changed)
        self.clahe_clip.valueChanged.connect(self._on_params_changed)
        self.clahe_grid.valueChanged.connect(self._on_params_changed)
        self.init_radius.valueChanged.connect(self._on_params_changed)
        self.growth_factor.valueChanged.connect(self._on_params_changed)
        self.orb_features.valueChanged.connect(self._on_params_changed)
        self.match_ratio.valueChanged.connect(self._on_params_changed)
        self.use_masked.toggled.connect(self._on_params_changed)
        self.reg_method.currentTextChanged.connect(self._on_reg_method_change)
        # Initialize visibility of ECC-specific controls
        self._on_reg_method_change(self.reg_method.currentText())

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
        self.seg_preview_btn = QPushButton("Preview Segmentation")
        # Initially disabled until a registration preview is successfully run
        self.seg_preview_btn.setEnabled(False)
        self.seg_preview_btn.clicked.connect(self._preview_segmentation)
        seg_form.addRow(self.seg_preview_btn)
        controls.addWidget(seg_group)
        self.seg_method.currentTextChanged.connect(self._persist_settings)
        self.invert.toggled.connect(self._persist_settings)
        self.manual_t.valueChanged.connect(self._persist_settings)
        self.adaptive_blk.valueChanged.connect(self._persist_settings)
        self.adaptive_C.valueChanged.connect(self._persist_settings)
        self.local_blk.valueChanged.connect(self._persist_settings)
        self.open_r.valueChanged.connect(self._persist_settings)
        self.close_r.valueChanged.connect(self._persist_settings)
        self.rm_obj.valueChanged.connect(self._persist_settings)
        self.rm_holes.valueChanged.connect(self._persist_settings)
        self.seg_method.currentTextChanged.connect(self._on_params_changed)
        self.invert.toggled.connect(self._on_params_changed)
        self.manual_t.valueChanged.connect(self._on_params_changed)
        self.adaptive_blk.valueChanged.connect(self._on_params_changed)
        self.adaptive_C.valueChanged.connect(self._on_params_changed)
        self.local_blk.valueChanged.connect(self._on_params_changed)
        self.open_r.valueChanged.connect(self._on_params_changed)
        self.close_r.valueChanged.connect(self._on_params_changed)
        self.rm_obj.valueChanged.connect(self._on_params_changed)
        self.rm_holes.valueChanged.connect(self._on_params_changed)

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
        run_box.addWidget(run_btn)
        run_box.addWidget(QLabel("Moving frame"))
        self.mov_idx_spin = QSpinBox(); self.mov_idx_spin.setMinimum(0); self.mov_idx_spin.setMaximum(0)
        run_box.addWidget(self.mov_idx_spin)
        controls.addLayout(run_box)

        controls.addStretch(1)

        # Right: viewer
        right = QVBoxLayout()
        layout.addLayout(right, 1)

        self.view = pg.ImageView()
        # Label histogram and ROI plot axes for clarity
        hist = self.view.getHistogramWidget()
        if hist is not None and hasattr(hist, 'axis'):
            hist.axis.setLabel("Pixel intensity", color="#FFFFFF")
        roi_plot = self.view.getRoiPlot()
        if roi_plot is not None:
            roi_plot.getAxis('bottom').setLabel("Frame", color="#FFFFFF")
            roi_plot.getAxis('left').setLabel("Mean intensity", color="#FFFFFF")
            roi_plot.showAxis('bottom', True)
            roi_plot.showAxis('left', True)
        right.addWidget(self.view)

        overlay_box = QHBoxLayout()
        self.overlay_ref_cb = QCheckBox("Show reference overlay"); self.overlay_ref_cb.setChecked(self.app.show_ref_overlay)
        self.overlay_mov_cb = QCheckBox("Show moving overlay"); self.overlay_mov_cb.setChecked(self.app.show_mov_overlay)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal); self.alpha_slider.setRange(0,100); self.alpha_slider.setValue(self.app.overlay_opacity)
        overlay_box.addWidget(self.overlay_ref_cb)
        overlay_box.addWidget(self.overlay_mov_cb)
        overlay_box.addWidget(QLabel("Opacity"))
        overlay_box.addWidget(self.alpha_slider)
        right.addLayout(overlay_box)

        mode_box = QHBoxLayout()
        mode_box.addWidget(QLabel("Mode"))
        self.overlay_mode_combo = QComboBox(); self.overlay_mode_combo.addItems(["magenta-green", "grayscale", "custom"])
        self.overlay_mode_combo.setCurrentText(self.app.overlay_mode)
        mode_box.addWidget(self.overlay_mode_combo)
        self.ref_color_label = QLabel("Ref")
        self.ref_color_btn = QPushButton(); self.ref_color_btn.setFixedWidth(30)
        self._set_btn_color(self.ref_color_btn, self.ref_color)
        self.mov_color_label = QLabel("Mov")
        self.mov_color_btn = QPushButton(); self.mov_color_btn.setFixedWidth(30)
        self._set_btn_color(self.mov_color_btn, self.mov_color)
        mode_box.addWidget(self.ref_color_label)
        mode_box.addWidget(self.ref_color_btn)
        mode_box.addWidget(self.mov_color_label)
        mode_box.addWidget(self.mov_color_btn)
        right.addLayout(mode_box)

        # Refresh overlays when controls change
        self.alpha_slider.valueChanged.connect(self._refresh_overlay_alpha)
        self.overlay_ref_cb.toggled.connect(self._refresh_overlay_alpha)
        self.overlay_mov_cb.toggled.connect(self._refresh_overlay_alpha)
        self.alpha_slider.valueChanged.connect(self._persist_settings)
        self.overlay_ref_cb.toggled.connect(self._persist_settings)
        self.overlay_mov_cb.toggled.connect(self._persist_settings)
        self.overlay_mode_combo.currentTextChanged.connect(self._on_overlay_mode_changed)
        self.overlay_mode_combo.currentTextChanged.connect(self._persist_settings)
        self.ref_color_btn.clicked.connect(lambda: self._choose_color('ref'))
        self.mov_color_btn.clicked.connect(lambda: self._choose_color('mov'))

        self._on_overlay_mode_changed(self.overlay_mode_combo.currentText())

        # Status
        self.status_label = QLabel("Ready.")
        right.addWidget(self.status_label)

        if self.app.last_folder:
            p = Path(self.app.last_folder)
            self.paths = discover_images(p)
            if self.paths:
                self.mov_idx_spin.setMaximum(len(self.paths) - 1)
                idx = self._show_reference_frame()
                if idx is not None:
                    self.status_label.setText(
                        f"Found {len(self.paths)} images. Preview: {self.paths[idx].name}")

    def _show_reference_frame(self):
        """Display the frame chosen by the current reference settings."""
        if not self.paths:
            return None
        choice = self.dir_combo.currentText()
        if choice == "first-to-last":
            idx = 0
        else:
            idx = len(self.paths) - 1
        idx = max(0, min(idx, len(self.paths) - 1))
        img = imread_gray(self.paths[idx], normalize=self.app.normalize,
                          scale_minmax=self.app.scale_minmax)
        self.view.setImage(img.T)
        return idx

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select image folder", "")
        if d:
            self.folder_edit.setText(d)
            # Persist the newly selected folder alongside other parameters
            self._persist_settings()
            # Changing folders invalidates previous registration
            self._registration_done = False
            self.seg_preview_btn.setEnabled(False)
            self.paths = discover_images(Path(d))
            if not self.paths:
                QMessageBox.warning(self, "No images", "No images found.")
                return
            self.mov_idx_spin.setMaximum(len(self.paths) - 1)
            idx = self._show_reference_frame()
            if idx is not None:
                self.status_label.setText(
                    f"Found {len(self.paths)} images. Preview: {self.paths[idx].name}")

    def _auto_scale_minmax(self):
        if not self.paths:
            QMessageBox.warning(self, "No images", "Choose an image folder first.")
            return
        lo, hi = compute_global_minmax(self.paths)
        self.scale_min.setValue(int(lo))
        self.scale_max.setValue(int(hi))
        self.status_label.setText(f"Global min/max: {lo}–{hi}")
        self._persist_settings()

    def _collect_params(self):
        # Pull from UI into dataclasses
        reg = RegParams(method=self.reg_method.currentText(),
                        model=self.reg_model.currentText(),
                        max_iters=self.max_iters.value(),
                        eps=self.eps.value(),
                        gauss_blur_sigma=self.gauss_sigma.value(),
                        clahe_clip=self.clahe_clip.value(),
                        clahe_grid=self.clahe_grid.value(),
                        initial_radius=self.init_radius.value(),
                        growth_factor=self.growth_factor.value(),
                        use_masked_ecc=self.use_masked.isChecked(),
                        orb_features=self.orb_features.value(),
                        match_ratio=self.match_ratio.value())
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
        scale_minmax = (self.scale_min.value(), self.scale_max.value())
        if scale_minmax[1] <= scale_minmax[0]:
            scale_minmax = None
        app = AppParams(direction=self.dir_combo.currentText(),
                        minutes_between_frames=self.dt_min.value(),
                        use_file_timestamps=self.use_ts.isChecked(),
                        normalize=self.norm_cb.isChecked(),
                        rescale_background=self.rescale_cb.isChecked(),
                        scale_minmax=scale_minmax,
                        show_ref_overlay=self.overlay_ref_cb.isChecked(),
                        show_mov_overlay=self.overlay_mov_cb.isChecked(),
                        overlay_opacity=self.alpha_slider.value(),
                        overlay_mode=self.overlay_mode_combo.currentText(),
                        overlay_ref_color=self.ref_color,
                        overlay_mov_color=self.mov_color)
        app.presets_path = self.app.presets_path
        return reg, seg, app

    def _persist_settings(self, *args):
        """Collect current UI state and save via QSettings."""
        last = self.folder_edit.text() or self.app.last_folder
        reg, seg, app = self._collect_params()
        app.last_folder = last
        self.reg, self.seg, self.app = reg, seg, app
        save_settings(reg, seg, app)
        return reg, seg, app

    def _save_preset(self):
        reg, seg, _ = self._persist_settings()
        initial = str(Path(self.app.presets_path) / "preset.json") if self.app.presets_path else "preset.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save Preset", initial, "JSON (*.json)")
        if path:
            self.app.presets_path = str(Path(path).parent)
            save_preset(path, reg, seg, self.app)
            save_settings(self.reg, self.seg, self.app)
            self.status_label.setText(f"Preset saved: {path}")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset", self.app.presets_path or "", "JSON (*.json)")
        if not path: return
        reg, seg, app = load_preset(path)
        app.presets_path = str(Path(path).parent)
        self.reg = reg; self.seg = seg; self.app = app
        # Update UI
        self.reg_method.setCurrentText(reg.method)
        self.reg_model.setCurrentText(reg.model)
        self.max_iters.setValue(reg.max_iters)
        self.eps.setValue(reg.eps)
        self.gauss_sigma.setValue(reg.gauss_blur_sigma)
        self.clahe_clip.setValue(reg.clahe_clip)
        self.clahe_grid.setValue(reg.clahe_grid)
        self.init_radius.setValue(reg.initial_radius)
        self.growth_factor.setValue(reg.growth_factor)
        self.use_masked.setChecked(reg.use_masked_ecc)
        self.orb_features.setValue(reg.orb_features)
        self.match_ratio.setValue(reg.match_ratio)
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
        self.dir_combo.setCurrentText(app.direction)
        self.dt_min.setValue(app.minutes_between_frames)
        self.use_ts.setChecked(app.use_file_timestamps)
        self.norm_cb.setChecked(app.normalize)
        if app.scale_minmax is not None:
            self.scale_min.setValue(int(app.scale_minmax[0]))
            self.scale_max.setValue(int(app.scale_minmax[1]))
        else:
            self.scale_min.setValue(0)
            self.scale_max.setValue(0)
        self.scale_min.setEnabled(self.norm_cb.isChecked())
        self.scale_max.setEnabled(self.norm_cb.isChecked())
        self.overlay_ref_cb.setChecked(app.show_ref_overlay)
        self.overlay_mov_cb.setChecked(app.show_mov_overlay)
        self.alpha_slider.setValue(app.overlay_opacity)
        self.overlay_mode_combo.setCurrentText(app.overlay_mode)
        self.ref_color = tuple(app.overlay_ref_color)
        self.mov_color = tuple(app.overlay_mov_color)
        self._set_btn_color(self.ref_color_btn, self.ref_color)
        self._set_btn_color(self.mov_color_btn, self.mov_color)
        self._on_overlay_mode_changed(app.overlay_mode)
        self.status_label.setText(f"Preset loaded: {path}")
        self._persist_settings()
        self._on_reg_method_change(self.reg_method.currentText())

    def _on_reg_method_change(self, method: str):
        """Enable or hide ECC-specific controls depending on method."""
        is_ecc = method == "ECC"
        is_orb = method == "ORB"
        self.max_iters.setEnabled(is_ecc)
        self.eps.setEnabled(is_ecc)
        self.max_iters.setVisible(is_ecc)
        self.eps.setVisible(is_ecc)
        self.max_iters_label.setVisible(is_ecc)
        self.eps_label.setVisible(is_ecc)
        self.orb_features.setEnabled(is_orb)
        self.match_ratio.setEnabled(is_orb)
        self.orb_features.setVisible(is_orb)
        self.match_ratio.setVisible(is_orb)
        self.orb_features_label.setVisible(is_orb)
        self.match_ratio_label.setVisible(is_orb)

    def _on_params_changed(self, *args):
        """Debounce rapid param changes and rerun active preview."""
        sender = self.sender()
        if sender is not None and hasattr(sender, "isEnabled") and not sender.isEnabled():
            return
        if self._current_preview not in ("registration", "segmentation"):
            return
        self._param_timer.start()

    def _apply_param_change(self):
        if self._current_preview == "registration":
            self._preview_registration()
        elif self._current_preview == "segmentation":
            self._preview_segmentation()

    def _refresh_overlay_alpha(self):
        """Blend cached overlays according to slider and checkbox states."""
        alpha = self.alpha_slider.value() / 100.0
        if self._current_preview == "registration" and self._reg_ref is not None and self._reg_warp is not None:
            mode = self.overlay_mode_combo.currentText()
            if mode == "magenta-green":
                ref_c = np.array([0, 255, 0])
                mov_c = np.array([255, 0, 255])
            elif mode == "grayscale":
                ref_c = mov_c = np.array([255, 255, 255])
            else:
                ref_c = np.array(self.ref_color)
                mov_c = np.array(self.mov_color)
            ref_color = cv2.cvtColor(self._reg_ref, cv2.COLOR_GRAY2RGB)
            mov_color = cv2.cvtColor(self._reg_warp, cv2.COLOR_GRAY2RGB)
            ref_color = (ref_color * (ref_c / 255)).astype(np.uint8)
            mov_color = (mov_color * (mov_c / 255)).astype(np.uint8)

            if self.overlay_ref_cb.isChecked() and self.overlay_mov_cb.isChecked():
                blend = cv2.addWeighted(ref_color, 1 - alpha, mov_color, alpha, 0)
            elif self.overlay_ref_cb.isChecked():
                blend = ref_color
            elif self.overlay_mov_cb.isChecked():
                blend = mov_color
            else:
                blend = np.zeros_like(ref_color)
            self.view.setImage(blend.transpose(1, 0, 2))
        elif self._current_preview == "segmentation" and self._seg_gray is not None and self._seg_overlay is not None:
            if self.overlay_ref_cb.isChecked() and self.overlay_mov_cb.isChecked():
                blend = cv2.addWeighted(self._seg_gray, 1 - alpha, self._seg_overlay, alpha, 0)
            elif self.overlay_ref_cb.isChecked():
                blend = self._seg_gray
            elif self.overlay_mov_cb.isChecked():
                blend = self._seg_overlay
            else:
                blend = np.zeros_like(self._seg_gray)
            self.view.setImage(blend.transpose(1, 0, 2))

    def _preview_registration(self):
        # Clear any previous previews so stale images aren't blended
        self._current_preview = None
        self._reg_ref = None
        self._reg_warp = None
        self._seg_gray = None
        self._seg_overlay = None
        # Reset registration flag until this preview completes successfully
        self._registration_done = False
        self.seg_preview_btn.setEnabled(False)

        if len(self.paths) < 2:
            QMessageBox.warning(self, "Need at least two images", "Load at least two images for preview.")
            return
        try:
            reg, _, app = self._persist_settings()
            if app.direction == "first-to-last":
                ref_idx = 0
            else:
                ref_idx = len(self.paths) - 1
            ref_img = imread_gray(self.paths[ref_idx], normalize=app.normalize,
                                 scale_minmax=app.scale_minmax)
            mov_idx = self.mov_idx_spin.value()
            if mov_idx < 0 or mov_idx >= len(self.paths):
                QMessageBox.warning(self, "Invalid index", "Select a valid moving frame index.")
                return
            mov_img = imread_gray(self.paths[mov_idx], normalize=app.normalize,
                                 scale_minmax=app.scale_minmax)
            ref_img = preprocess(ref_img, reg.gauss_blur_sigma, reg.clahe_clip, reg.clahe_grid)
            mov_img = preprocess(mov_img, reg.gauss_blur_sigma, reg.clahe_clip, reg.clahe_grid)
            if reg.method.upper() == "ORB":
                _, warped, _ = register_orb(ref_img, mov_img, model=reg.model,
                                            orb_features=reg.orb_features,
                                            match_ratio=reg.match_ratio)
            else:
                _, warped, _ = register_ecc(ref_img, mov_img, model=reg.model,
                                            max_iters=reg.max_iters, eps=reg.eps)
            self._reg_ref = ref_img
            self._reg_warp = warped
            self._current_preview = "registration"
            # Blend and push the new image to the viewer
            self._refresh_overlay_alpha()
            self.view.setImage(self.view.imageItem.image)
            self.status_label.setText("Preview successful.")
            # Enable segmentation preview now that registration succeeded
            self._registration_done = True
            self.seg_preview_btn.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")

    def _preview_segmentation(self):
        # Clear any previous previews so stale images aren't blended
        self._current_preview = None
        self._seg_gray = None
        self._seg_overlay = None

        if not self.paths:
            QMessageBox.warning(self, "No images", "Choose an image folder first.")
            return
        try:
            reg, seg, app = self._persist_settings()

            # Determine reference and moving indices
            if app.direction == "first-to-last":
                ref_idx = 0
            else:
                ref_idx = len(self.paths) - 1

            mov_idx = self.mov_idx_spin.value()
            if mov_idx < 0 or mov_idx >= len(self.paths):
                QMessageBox.warning(self, "Invalid index", "Select a valid moving frame index.")
                return

            # Reuse previously registered images or run a quick registration
            if self._reg_ref is None or self._reg_warp is None:
                ref_img = imread_gray(self.paths[ref_idx], normalize=app.normalize,
                                     scale_minmax=app.scale_minmax)
                mov_img = imread_gray(self.paths[mov_idx], normalize=app.normalize,
                                     scale_minmax=app.scale_minmax)
                ref_img = preprocess(ref_img, reg.gauss_blur_sigma, reg.clahe_clip, reg.clahe_grid)
                mov_img = preprocess(mov_img, reg.gauss_blur_sigma, reg.clahe_clip, reg.clahe_grid)
                if reg.method.upper() == "ORB":
                    _, warped, _ = register_orb(ref_img, mov_img, model=reg.model,
                                                orb_features=reg.orb_features,
                                                match_ratio=reg.match_ratio)
                else:
                    _, warped, _ = register_ecc(ref_img, mov_img, model=reg.model,
                                                max_iters=reg.max_iters, eps=reg.eps)
                self._reg_ref = ref_img
                self._reg_warp = warped

            gray = self._reg_warp
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

            self._seg_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self._seg_overlay = cv2.cvtColor(overlay_outline(gray, bw), cv2.COLOR_BGR2RGB)
            self._current_preview = "segmentation"
            self._registration_done = True

            # Blend and push the new image to the viewer
            self._refresh_overlay_alpha()
            self.view.setImage(self.view.imageItem.image)
            self.status_label.setText("Segmentation preview successful.")
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")

    def _run_pipeline(self):
        if not self.paths:
            QMessageBox.warning(self, "No images", "Choose an image folder first.")
            return

        reg, seg, app = self._persist_settings()
        # Build slim dicts for worker
        reg_cfg = dict(method=reg.method, model=reg.model, max_iters=reg.max_iters,
                       eps=reg.eps, use_masked_ecc=reg.use_masked_ecc,
                       gauss_blur_sigma=reg.gauss_blur_sigma,
                       clahe_clip=reg.clahe_clip,
                       clahe_grid=reg.clahe_grid,
                       initial_radius=reg.initial_radius,
                       growth_factor=reg.growth_factor,
                       orb_features=reg.orb_features,
                       match_ratio=reg.match_ratio)
        seg_cfg = dict(method=seg.method, invert=seg.invert, manual_thresh=seg.manual_thresh,
                       adaptive_block=seg.adaptive_block, adaptive_C=seg.adaptive_C, local_block=seg.local_block,
                       morph_open_radius=seg.morph_open_radius, morph_close_radius=seg.morph_close_radius,
                       remove_objects_smaller_px=seg.remove_objects_smaller_px, remove_holes_smaller_px=seg.remove_holes_smaller_px)
        app_cfg = dict(direction=app.direction,
                       use_difference_for_seg=False, save_intermediates=True,
                       normalize=app.normalize,
                       rescale_background=app.rescale_background,
                       scale_minmax=app.scale_minmax)

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

    def closeEvent(self, event):
        """Persist current settings when the window is closed."""
        self._persist_settings()
        super().closeEvent(event)
