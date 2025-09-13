from __future__ import annotations
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QGroupBox,
    QFormLayout,
    QGridLayout,
    QLineEdit,
    QToolTip,
    QColorDialog,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QThread, QTimer
import logging
from pathlib import Path
import pyqtgraph as pg
import numpy as np
import cv2

from ..models.config import (
    RegParams,
    SegParams,
    AppParams,
    save_settings,
    load_settings,
    save_preset,
    load_preset,
)
from ..core.io_utils import (
    discover_images,
    imread_gray,
    compute_global_minmax,
)
from ..core.registration import register_ecc, register_orb, register_orb_ecc, preprocess
from ..core.segmentation import segment
from ..core.processing import overlay_outline
from ..core.difference import compute_difference
from ..workers.pipeline_worker import PipelineWorker
from .collapsible_section import CollapsibleSection

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yeast Flowcell Analyzer — PyQt")
        self.resize(1200, 800)
        self.reg, self.seg, self.app = load_settings()
        self.ref_color = tuple(self.app.overlay_ref_color)
        self.mov_color = tuple(self.app.overlay_mov_color)
        self.new_color = tuple(self.app.overlay_new_color)
        self.lost_color = tuple(self.app.overlay_lost_color)
        self.paths: list[Path] = []
        # Cached preview images for alpha blending
        self._reg_ref = None
        self._reg_warp = None
        self._seg_gray = None
        self._seg_overlay = None
        self._diff_img = None
        self._diff_gray = None
        self._current_preview = None
        self._build_ui()
        self._update_seg_controls(self.seg_method.currentText())
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
        logger.info("Color button clicked: %s", which)
        mapping = {
            "ref": self.ref_color,
            "mov": self.mov_color,
            "new": self.new_color,
            "lost": self.lost_color,
        }
        initial = mapping.get(which, (255, 255, 255))
        col = QColorDialog.getColor(QColor(*initial), self, "Select color")
        if col.isValid():
            color = (col.red(), col.green(), col.blue())
            logger.info("%s color selected: %s", which, color)
            if which == "ref":
                self.ref_color = color
                self._set_btn_color(self.ref_color_btn, color)
            elif which == "mov":
                self.mov_color = color
                self._set_btn_color(self.mov_color_btn, color)
            elif which == "new":
                self.new_color = color
                self._set_btn_color(self.new_color_btn, color)
            elif which == "lost":
                self.lost_color = color
                self._set_btn_color(self.lost_color_btn, color)
            self._refresh_overlay_alpha()
            self._persist_settings()
        else:
            logger.info("%s color selection canceled", which)

    def _on_overlay_mode_changed(self, mode: str) -> None:
        custom = mode == "custom"
        for w in (
            self.ref_color_btn,
            self.mov_color_btn,
            self.ref_color_label,
            self.mov_color_label,
            self.new_color_btn,
            self.lost_color_btn,
            self.new_color_label,
            self.lost_color_label,
        ):
            w.setVisible(custom)
        self._refresh_overlay_alpha()
        logger.info("Overlay mode changed: %s", mode)

    def _update_seg_controls(self, method: str) -> None:
        """Enable/disable threshold widgets based on segmentation method."""
        manual = method == "manual"
        adaptive = method == "adaptive"
        local = method == "local"
        self.manual_t.setEnabled(manual)
        self.adaptive_blk.setEnabled(adaptive)
        self.adaptive_C.setEnabled(adaptive)
        self.local_blk.setEnabled(local)

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
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(["last-to-first", "first-to-last"])
        self.dir_combo.setCurrentText(self.app.direction)
        self.use_ts = QCheckBox("Use file timestamps for frame spacing")
        self.use_ts.setChecked(self.app.use_file_timestamps)
        self.dt_min = QDoubleSpinBox()
        self.dt_min.setDecimals(3)
        self.dt_min.setMinimum(0.0)
        self.dt_min.setValue(self.app.minutes_between_frames)
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
        self.scale_min = QSpinBox()
        self.scale_min.setRange(-1000000, 1000000)
        self.scale_max = QSpinBox()
        self.scale_max.setRange(-1000000, 1000000)
        if self.app.scale_minmax is not None:
            self.scale_min.setValue(int(self.app.scale_minmax[0]))
            self.scale_max.setValue(int(self.app.scale_minmax[1]))
        else:
            self.scale_min.setValue(0)
            self.scale_max.setValue(0)
        self.bg_sub_cb = QCheckBox("Subtract background")
        self.bg_sub_cb.setChecked(self.app.subtract_background)
        intensity_form.addRow(self.norm_cb)
        intensity_form.addRow("Min", self.scale_min)
        intensity_form.addRow("Max", self.scale_max)
        self.auto_scale_btn = QPushButton("Auto min/max")
        self.auto_scale_btn.clicked.connect(self._auto_scale_minmax)
        intensity_form.addRow(self.auto_scale_btn)
        intensity_form.addRow(self.bg_sub_cb)
        controls.addWidget(intensity_group)
        self.scale_min.setEnabled(self.norm_cb.isChecked())
        self.scale_max.setEnabled(self.norm_cb.isChecked())
        self.norm_cb.toggled.connect(
            lambda v: [self.scale_min.setEnabled(v), self.scale_max.setEnabled(v)]
        )
        self.norm_cb.toggled.connect(self._persist_settings)
        self.scale_min.valueChanged.connect(self._persist_settings)
        self.scale_max.valueChanged.connect(self._persist_settings)
        self.bg_sub_cb.toggled.connect(self._persist_settings)
        self.norm_cb.toggled.connect(self._on_params_changed)
        self.scale_min.valueChanged.connect(self._on_params_changed)
        self.scale_max.valueChanged.connect(self._on_params_changed)
        self.bg_sub_cb.toggled.connect(self._on_params_changed)

        # Registration params
        reg_section = CollapsibleSection("Registration")
        reg_grid = QGridLayout()
        self.reg_method = QComboBox()
        self.reg_method.addItems(["ECC", "ORB", "ORB+ECC"])
        self.reg_method.setCurrentText(self.reg.method)
        self.reg_model = QComboBox()
        self.reg_model.addItems(["translation", "euclidean", "affine", "homography"])
        self.reg_model.setCurrentText(self.reg.model)
        self.max_iters_label = QLabel("Max iters")
        self.max_iters = QSpinBox()
        self.max_iters.setRange(10, 10000)
        self.max_iters.setValue(self.reg.max_iters)
        self.eps_label = QLabel("Epsilon")
        self.eps = QDoubleSpinBox()
        self.eps.setDecimals(9)
        self.eps.setSingleStep(1e-6)
        self.eps.setValue(self.reg.eps)
        gauss_label = QLabel("Gaussian σ")
        self.gauss_sigma = QDoubleSpinBox()
        self.gauss_sigma.setRange(0.0, 10.0)
        self.gauss_sigma.setDecimals(2)
        self.gauss_sigma.setSingleStep(0.1)
        self.gauss_sigma.setValue(self.reg.gauss_blur_sigma)
        self.use_clahe = QCheckBox("Use CLAHE")
        self.use_clahe.setChecked(self.reg.use_clahe)
        self.clahe_clip = QDoubleSpinBox()
        self.clahe_clip.setRange(0.0, 40.0)
        self.clahe_clip.setDecimals(2)
        self.clahe_clip.setSingleStep(0.1)
        self.clahe_clip.setValue(self.reg.clahe_clip)
        clahe_grid_label = QLabel("CLAHE grid")
        self.clahe_grid = QSpinBox()
        self.clahe_grid.setRange(1, 64)
        self.clahe_grid.setValue(self.reg.clahe_grid)
        self.init_radius_label = QLabel("Initial radius")
        self.init_radius = QSpinBox()
        self.init_radius.setRange(0, 1000)
        self.init_radius.setValue(self.reg.initial_radius)
        self.growth_factor_label = QLabel("Growth factor")
        self.growth_factor = QDoubleSpinBox()
        self.growth_factor.setRange(0.1, 10.0)
        self.growth_factor.setDecimals(2)
        self.growth_factor.setSingleStep(0.1)
        self.growth_factor.setValue(self.reg.growth_factor)
        self.use_masked_label = QLabel("Use masked ECC")
        self.use_masked = QCheckBox()
        self.use_masked.setChecked(self.reg.use_masked_ecc)
        self.orb_features_label = QLabel("ORB features")
        self.orb_features = QSpinBox()
        self.orb_features.setRange(1, 100000)
        self.orb_features.setValue(self.reg.orb_features)
        self.match_ratio_label = QLabel("Match ratio")
        self.match_ratio = QDoubleSpinBox()
        self.match_ratio.setRange(0.0, 1.0)
        self.match_ratio.setDecimals(2)
        self.match_ratio.setSingleStep(0.05)
        self.match_ratio.setValue(self.reg.match_ratio)
        self.use_ecc_label = QLabel("ECC fallback")
        self.use_ecc_fallback = QCheckBox()
        self.use_ecc_fallback.setChecked(self.reg.use_ecc_fallback)
        self.min_keypoints_label = QLabel("Min keypoints")
        self.min_keypoints = QSpinBox()
        self.min_keypoints.setRange(0, 100000)
        self.min_keypoints.setValue(self.reg.min_keypoints)
        self.min_matches_label = QLabel("Min matches")
        self.min_matches = QSpinBox()
        self.min_matches.setRange(0, 100000)
        self.min_matches.setValue(self.reg.min_matches)
        reg_grid.addWidget(QLabel("Method"), 0, 0)
        reg_grid.addWidget(self.reg_method, 0, 1)
        reg_grid.addWidget(QLabel("Model"), 1, 0)
        reg_grid.addWidget(self.reg_model, 1, 1)
        reg_grid.addWidget(self.max_iters_label, 2, 0)
        reg_grid.addWidget(self.max_iters, 2, 1)
        reg_grid.addWidget(self.eps_label, 3, 0)
        reg_grid.addWidget(self.eps, 3, 1)
        reg_grid.addWidget(gauss_label, 4, 0)
        reg_grid.addWidget(self.gauss_sigma, 4, 1)
        reg_grid.addWidget(self.use_clahe, 5, 0)
        reg_grid.addWidget(self.clahe_clip, 5, 1)
        reg_grid.addWidget(clahe_grid_label, 5, 2)
        reg_grid.addWidget(self.clahe_grid, 5, 3)
        reg_grid.addWidget(self.init_radius_label, 1, 2)
        reg_grid.addWidget(self.init_radius, 1, 3)
        reg_grid.addWidget(self.growth_factor_label, 2, 2)
        reg_grid.addWidget(self.growth_factor, 2, 3)
        reg_grid.addWidget(self.use_masked_label, 3, 2)
        reg_grid.addWidget(self.use_masked, 3, 3)
        reg_grid.addWidget(self.orb_features_label, 4, 2)
        reg_grid.addWidget(self.orb_features, 4, 3)
        reg_grid.addWidget(self.match_ratio_label, 6, 2)
        reg_grid.addWidget(self.match_ratio, 6, 3)
        reg_grid.addWidget(self.use_ecc_label, 7, 0)
        reg_grid.addWidget(self.use_ecc_fallback, 7, 1)
        reg_grid.addWidget(self.min_keypoints_label, 7, 2)
        reg_grid.addWidget(self.min_keypoints, 7, 3)
        reg_grid.addWidget(self.min_matches_label, 8, 2)
        reg_grid.addWidget(self.min_matches, 8, 3)
        self.kp_label = QLabel("Keypoints: -")
        reg_grid.addWidget(self.kp_label, 9, 0, 1, 4)
        self.clahe_clip.setEnabled(self.use_clahe.isChecked())
        self.clahe_grid.setEnabled(self.use_clahe.isChecked())
        self.use_clahe.toggled.connect(
            lambda v: [self.clahe_clip.setEnabled(v), self.clahe_grid.setEnabled(v)]
        )
        reg_section.setContentLayout(reg_grid)
        controls.addWidget(reg_section)
        reg_preview_btn = QPushButton("Preview Registration")
        reg_preview_btn.clicked.connect(self._preview_registration)
        controls.addWidget(reg_preview_btn)
        self._add_help(
            self.reg_method,
            "Registration algorithm. ECC correlates intensities for higher accuracy but "
            "is slower; ORB matches keypoints for speed and robustness to large "
            "motions.",
        )
        self._add_help(
            self.reg_model,
            "Geometric transform model. Translation is fastest; Euclidean adds rotation; "
            "Affine adds shear/scale; Homography handles perspective but is slowest.",
        )
        self._add_help(
            self.max_iters,
            "Maximum ECC iterations. More iterations improve alignment but slow "
            "processing. Typical range: 50–300.",
        )
        self._add_help(
            self.eps,
            "ECC convergence threshold. Smaller values yield more precise results at "
            "the cost of extra iterations. Recommended: 1e-4–1e-6.",
        )
        self._add_help(
            self.gauss_sigma,
            "Gaussian blur σ before registration to reduce noise. 0–2 is common; "
            "higher values smooth detail (faster, less accurate).",
        )
        self._add_help(
            self.use_clahe,
            "Enable CLAHE local contrast enhancement before registration.",
        )
        self._add_help(
            self.clahe_clip,
            "CLAHE clip limit for local contrast enhancement. 0 disables. Typical "
            "range: 0–5; higher improves contrast but may amplify noise.",
        )
        self._add_help(
            self.clahe_grid,
            "CLAHE tile grid size. Smaller (8–16) boosts local detail but may add "
            "artifacts; larger (up to 32) is smoother but less adaptive.",
        )
        self._add_help(
            self.init_radius, "Initial search window radius in pixels for ECC."
        )
        self._add_help(
            self.growth_factor,
            "Scale search window after each registration step (>=1 keeps more context).",
        )
        self._add_help(self.orb_features, "Number of ORB features to detect.")
        self._add_help(self.match_ratio, "Lowe ratio for filtering ORB matches.")
        self._add_help(self.use_ecc_fallback, "Fallback to ECC when ORB fails.")
        self._add_help(
            self.min_keypoints, "Minimum detected ORB keypoints to attempt matching."
        )
        self._add_help(
            self.min_matches, "Minimum good ORB matches before estimating transform."
        )
        self._add_help(
            self.use_masked,
            "Use segmentation mask during ECC to focus on cells, improving "
            "accuracy in cluttered scenes but requiring prior segmentation.",
        )
        self.reg_method.currentTextChanged.connect(self._persist_settings)
        self.reg_model.currentTextChanged.connect(self._persist_settings)
        self.max_iters.valueChanged.connect(self._persist_settings)
        self.eps.valueChanged.connect(self._persist_settings)
        self.gauss_sigma.valueChanged.connect(self._persist_settings)
        self.use_clahe.toggled.connect(self._persist_settings)
        self.clahe_clip.valueChanged.connect(self._persist_settings)
        self.clahe_grid.valueChanged.connect(self._persist_settings)
        self.init_radius.valueChanged.connect(self._persist_settings)
        self.growth_factor.valueChanged.connect(self._persist_settings)
        self.orb_features.valueChanged.connect(self._persist_settings)
        self.match_ratio.valueChanged.connect(self._persist_settings)
        self.use_ecc_fallback.toggled.connect(self._persist_settings)
        self.min_keypoints.valueChanged.connect(self._persist_settings)
        self.min_matches.valueChanged.connect(self._persist_settings)
        self.use_masked.toggled.connect(self._persist_settings)
        self.reg_method.currentTextChanged.connect(self._on_params_changed)
        self.reg_model.currentTextChanged.connect(self._on_params_changed)
        self.max_iters.valueChanged.connect(self._on_params_changed)
        self.eps.valueChanged.connect(self._on_params_changed)
        self.gauss_sigma.valueChanged.connect(self._on_params_changed)
        self.use_clahe.toggled.connect(self._on_params_changed)
        self.clahe_clip.valueChanged.connect(self._on_params_changed)
        self.clahe_grid.valueChanged.connect(self._on_params_changed)
        self.init_radius.valueChanged.connect(self._on_params_changed)
        self.growth_factor.valueChanged.connect(self._on_params_changed)
        self.orb_features.valueChanged.connect(self._on_params_changed)
        self.match_ratio.valueChanged.connect(self._on_params_changed)
        self.use_ecc_fallback.toggled.connect(self._on_params_changed)
        self.min_keypoints.valueChanged.connect(self._on_params_changed)
        self.min_matches.valueChanged.connect(self._on_params_changed)
        self.use_masked.toggled.connect(self._on_params_changed)
        self.reg_method.currentTextChanged.connect(self._on_reg_method_change)
        # Initialize visibility of ECC-specific controls
        self._on_reg_method_change(self.reg_method.currentText())

        # Difference preview
        diff_section = CollapsibleSection("Difference")
        diff_layout = QVBoxLayout()
        self.diff_method = QComboBox()
        self.diff_method.addItems(["abs", "lab", "edges"])
        self.diff_method.setCurrentText(self.app.difference_method)
        diff_layout.addWidget(self.diff_method)
        diff_section.setContentLayout(diff_layout)
        controls.addWidget(diff_section)
        self.diff_preview_btn = QPushButton("Preview Difference")
        self.diff_preview_btn.setEnabled(False)
        self.diff_preview_btn.clicked.connect(self._preview_difference)
        controls.addWidget(self.diff_preview_btn)
        self.diff_method.currentTextChanged.connect(self._persist_settings)
        self.diff_method.currentTextChanged.connect(self._on_params_changed)

        # Segmentation params
        seg_section = CollapsibleSection("Segmentation")
        seg_grid = QGridLayout()
        self.seg_method = QComboBox()
        self.seg_method.addItems(
            [
                "otsu",
                "multi_otsu",
                "li",
                "yen",
                "adaptive",
                "local",
                "manual",
            ]
        )
        self.seg_method.setCurrentText(self.seg.method)
        self.invert = QCheckBox("Cells darker (invert)")
        self.invert.setChecked(self.seg.invert)
        self.skip_outline = QCheckBox("Skip outline prefilter")
        self.skip_outline.setChecked(self.seg.skip_outline)
        self.manual_t = QSpinBox()
        self.manual_t.setRange(0, 255)
        self.manual_t.setValue(self.seg.manual_thresh)
        self.adaptive_blk = QSpinBox()
        self.adaptive_blk.setRange(3, 999)
        self.adaptive_blk.setSingleStep(2)
        self.adaptive_blk.setValue(self.seg.adaptive_block)
        self.adaptive_C = QSpinBox()
        self.adaptive_C.setRange(-100, 100)
        self.adaptive_C.setValue(self.seg.adaptive_C)
        self.local_blk = QSpinBox()
        self.local_blk.setRange(3, 999)
        self.local_blk.setSingleStep(2)
        self.local_blk.setValue(self.seg.local_block)
        self.open_r = QSpinBox()
        self.open_r.setRange(0, 50)
        if self.seg.morph_open_radius is not None:
            self.open_r.setValue(self.seg.morph_open_radius)
        else:
            self.open_r.lineEdit().clear()
        self.close_r = QSpinBox()
        self.close_r.setRange(0, 50)
        if self.seg.morph_close_radius is not None:
            self.close_r.setValue(self.seg.morph_close_radius)
        else:
            self.close_r.lineEdit().clear()
        self.rm_obj = QSpinBox()
        self.rm_obj.setRange(0, 100000)
        self.rm_obj.setValue(self.seg.remove_objects_smaller_px)
        self.rm_holes = QSpinBox()
        self.rm_holes.setRange(0, 100000)
        self.rm_holes.setValue(self.seg.remove_holes_smaller_px)
        seg_grid.addWidget(QLabel("Method"), 0, 0)
        seg_grid.addWidget(self.seg_method, 0, 1)
        seg_grid.addWidget(self.invert, 1, 0, 1, 2)
        seg_grid.addWidget(self.skip_outline, 2, 0, 1, 2)
        seg_grid.addWidget(QLabel("Manual threshold"), 3, 0)
        seg_grid.addWidget(self.manual_t, 3, 1)
        seg_grid.addWidget(QLabel("Adaptive block"), 4, 0)
        seg_grid.addWidget(self.adaptive_blk, 4, 1)
        seg_grid.addWidget(QLabel("Adaptive C"), 5, 0)
        seg_grid.addWidget(self.adaptive_C, 5, 1)
        seg_grid.addWidget(QLabel("Local block"), 0, 2)
        seg_grid.addWidget(self.local_blk, 0, 3)
        seg_grid.addWidget(QLabel("Open radius"), 1, 2)
        seg_grid.addWidget(self.open_r, 1, 3)
        seg_grid.addWidget(QLabel("Close radius"), 2, 2)
        seg_grid.addWidget(self.close_r, 2, 3)
        seg_grid.addWidget(QLabel("Remove objects < px"), 3, 2)
        seg_grid.addWidget(self.rm_obj, 3, 3)
        seg_grid.addWidget(QLabel("Remove holes < px"), 4, 2)
        seg_grid.addWidget(self.rm_holes, 4, 3)
        seg_section.setContentLayout(seg_grid)
        controls.addWidget(seg_section)
        self.seg_preview_btn = QPushButton("Preview Segmentation")
        # Initially disabled until a registration preview is successfully run
        self.seg_preview_btn.setEnabled(False)
        self.seg_preview_btn.clicked.connect(self._preview_segmentation)
        controls.addWidget(self.seg_preview_btn)
        self._add_help(
            self.seg_method,
            "Segmentation algorithm. Otsu chooses a global threshold; Adaptive and Local use"
            " local statistics; Manual applies a fixed threshold.",
        )
        self._add_help(
            self.invert,
            "Check when cells appear darker than background to invert intensities before"
            " thresholding.",
        )
        self._add_help(
            self.skip_outline,
            "Skip preprocessing that enhances cell outlines. Speeds up processing but may"
            " reduce accuracy; automatically skipped for difference images or low contrast.",
        )
        self._add_help(
            self.manual_t,
            "Manual global threshold (0–255). Pixels above become foreground when method is manual."
            " Typical range 100–160.",
        )
        self._add_help(
            self.adaptive_blk,
            "Odd block size for adaptive thresholding—31–151 (odd) recommended; larger values"
            " smooth over more area while smaller values preserve detail.",
        )
        self._add_help(
            self.adaptive_C,
            "Constant subtracted in adaptive method—positive values make segmentation more"
            " selective. Typical range −10 to 10.",
        )
        self._add_help(
            self.local_blk,
            "Odd block size for local thresholding from scikit-image—31–151 (odd) recommended;"
            " sets the neighborhood used to compute thresholds.",
        )
        self._add_help(
            self.open_r,
            "Radius for morphological opening to remove small bright specks—0–5 px recommended.",
        )
        self._add_help(
            self.close_r,
            "Radius for morphological closing to fill small dark gaps—0–5 px recommended.",
        )
        self._add_help(
            self.rm_obj,
            "Remove connected components smaller than this area in pixels to discard noise—typical"
            " range 0–1000 px².",
        )
        self._add_help(
            self.rm_holes,
            "Fill holes smaller than this area in pixels within segmented objects—typical range"
            " 0–1000 px².",
        )
        self.seg_method.currentTextChanged.connect(self._persist_settings)
        self.seg_method.currentTextChanged.connect(self._update_seg_controls)
        self.invert.toggled.connect(self._persist_settings)
        self.skip_outline.toggled.connect(self._persist_settings)
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
        self.skip_outline.toggled.connect(self._on_params_changed)
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
        save_p = QPushButton("Save Preset")
        load_p = QPushButton("Load Preset")
        save_p.clicked.connect(self._save_preset)
        load_p.clicked.connect(self._load_preset)
        preset_box.addWidget(save_p)
        preset_box.addWidget(load_p)
        controls.addLayout(preset_box)

        # Run / Preview
        run_box = QHBoxLayout()
        run_btn = QPushButton("Run Analysis")
        run_btn.clicked.connect(self._run_pipeline)
        run_box.addWidget(run_btn)
        run_box.addWidget(QLabel("Frame pair"))
        self.mov_idx_spin = QSpinBox()
        self.mov_idx_spin.setMinimum(0)
        self.mov_idx_spin.setMaximum(0)
        self.mov_idx_spin.setToolTip("Select frame pair index")
        run_box.addWidget(self.mov_idx_spin)
        controls.addLayout(run_box)

        self.save_intermediates = QCheckBox("Save intermediate images")
        self.save_intermediates.setChecked(self.app.save_intermediates)
        controls.addWidget(self.save_intermediates)
        self.save_intermediates.toggled.connect(self._persist_settings)

        self.archive_intermediates = QCheckBox("Archive/delete intermediate images after run")
        self.archive_intermediates.setChecked(self.app.archive_intermediates)
        controls.addWidget(self.archive_intermediates)
        self.archive_intermediates.toggled.connect(self._persist_settings)

        self.save_masks_checkbox = QCheckBox("Save difference masks")
        self.save_masks_checkbox.setChecked(self.app.save_masks)
        controls.addWidget(self.save_masks_checkbox)
        self.save_masks_checkbox.toggled.connect(self._persist_settings)

        self.save_gm_checkbox = QCheckBox("Save GM composites")
        self.save_gm_checkbox.setChecked(self.app.save_gm_composite)
        controls.addWidget(self.save_gm_checkbox)
        self.save_gm_checkbox.toggled.connect(self._persist_settings)

        controls.addStretch(1)

        # Right: viewer
        right = QVBoxLayout()
        layout.addLayout(right, 1)

        self.view = pg.ImageView()
        # Label histogram and ROI plot axes for clarity
        hist = self.view.getHistogramWidget()
        if hist is not None and hasattr(hist, "axis"):
            hist.axis.setLabel("Pixel intensity", color="#FFFFFF")
        roi_plot = self.view.getRoiPlot()
        if roi_plot is not None:
            roi_plot.getAxis("bottom").setLabel("Frame", color="#FFFFFF")
            roi_plot.getAxis("left").setLabel("Mean intensity", color="#FFFFFF")
            roi_plot.showAxis("bottom", True)
            roi_plot.showAxis("left", True)
        right.addWidget(self.view)

        overlay_box = QHBoxLayout()
        self.overlay_ref_cb = QCheckBox("Show reference overlay")
        self.overlay_ref_cb.setChecked(self.app.show_ref_overlay)
        self.overlay_mov_cb = QCheckBox("Show moving overlay")
        self.overlay_mov_cb.setChecked(self.app.show_mov_overlay)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(self.app.overlay_opacity)
        overlay_box.addWidget(self.overlay_ref_cb)
        overlay_box.addWidget(self.overlay_mov_cb)
        overlay_box.addWidget(QLabel("Opacity"))
        overlay_box.addWidget(self.alpha_slider)
        right.addLayout(overlay_box)

        mode_box = QHBoxLayout()
        mode_box.addWidget(QLabel("Mode"))
        self.overlay_mode_combo = QComboBox()
        self.overlay_mode_combo.addItems(["magenta-green", "grayscale", "custom"])
        self.overlay_mode_combo.setCurrentText(self.app.overlay_mode)
        mode_box.addWidget(self.overlay_mode_combo)
        self.ref_color_label = QLabel("Ref")
        self.ref_color_btn = QPushButton()
        self.ref_color_btn.setFixedWidth(30)
        self._set_btn_color(self.ref_color_btn, self.ref_color)
        self.mov_color_label = QLabel("Mov")
        self.mov_color_btn = QPushButton()
        self.mov_color_btn.setFixedWidth(30)
        self._set_btn_color(self.mov_color_btn, self.mov_color)
        self.new_color_label = QLabel("New")
        self.new_color_btn = QPushButton()
        self.new_color_btn.setFixedWidth(30)
        self._set_btn_color(self.new_color_btn, self.new_color)
        self.lost_color_label = QLabel("Lost")
        self.lost_color_btn = QPushButton()
        self.lost_color_btn.setFixedWidth(30)
        self._set_btn_color(self.lost_color_btn, self.lost_color)
        mode_box.addWidget(self.ref_color_label)
        mode_box.addWidget(self.ref_color_btn)
        mode_box.addWidget(self.mov_color_label)
        mode_box.addWidget(self.mov_color_btn)
        mode_box.addWidget(self.new_color_label)
        mode_box.addWidget(self.new_color_btn)
        mode_box.addWidget(self.lost_color_label)
        mode_box.addWidget(self.lost_color_btn)
        right.addLayout(mode_box)

        # Refresh overlays when controls change
        self.alpha_slider.valueChanged.connect(self._refresh_overlay_alpha)
        self.overlay_ref_cb.toggled.connect(self._refresh_overlay_alpha)
        self.overlay_mov_cb.toggled.connect(self._refresh_overlay_alpha)
        self.alpha_slider.valueChanged.connect(self._persist_settings)
        self.overlay_ref_cb.toggled.connect(self._persist_settings)
        self.overlay_mov_cb.toggled.connect(self._persist_settings)
        self.overlay_mode_combo.currentTextChanged.connect(
            self._on_overlay_mode_changed
        )
        self.overlay_mode_combo.currentTextChanged.connect(self._persist_settings)
        self.ref_color_btn.clicked.connect(lambda: self._choose_color("ref"))
        self.mov_color_btn.clicked.connect(lambda: self._choose_color("mov"))
        self.new_color_btn.clicked.connect(lambda: self._choose_color("new"))
        self.lost_color_btn.clicked.connect(lambda: self._choose_color("lost"))

        self._on_overlay_mode_changed(self.overlay_mode_combo.currentText())

        # Status
        self.status_label = QLabel("Ready.")
        right.addWidget(self.status_label)

        if self.app.last_folder:
            p = Path(self.app.last_folder)
            self.paths = discover_images(p)
            if self.paths:
                self.mov_idx_spin.setMaximum(max(0, len(self.paths) - 2))
                self.mov_idx_spin.setValue(0)
                idx = self._show_reference_frame()
                if idx is not None:
                    self.status_label.setText(
                        f"Found {len(self.paths)} images. Preview: {self.paths[idx].name}"
                    )
        logger.info("UI build complete")

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
        img = imread_gray(
            self.paths[idx],
            normalize=self.app.normalize,
            scale_minmax=self.app.scale_minmax,
        )
        self.view.setImage(img.T)
        return idx

    def _choose_folder(self):
        logger.info("Browse button clicked")
        d = QFileDialog.getExistingDirectory(self, "Select image folder", "")
        if d:
            logger.info("Folder selected: %s", d)
            self.folder_edit.setText(d)
            # Persist the newly selected folder alongside other parameters
            self._persist_settings()
            # Changing folders invalidates previous registration
            self._registration_done = False
            self.seg_preview_btn.setEnabled(False)
            self.diff_preview_btn.setEnabled(False)
            self._reg_ref = None
            self._reg_warp = None
            self._seg_gray = None
            self._seg_overlay = None
            self._diff_img = None
            self._diff_gray = None
            self.paths = discover_images(Path(d))
            if not self.paths:
                QMessageBox.warning(self, "No images", "No images found.")
                return
            self.mov_idx_spin.setMaximum(max(0, len(self.paths) - 2))
            self.mov_idx_spin.setValue(0)
            idx = self._show_reference_frame()
            if idx is not None:
                self.status_label.setText(
                    f"Found {len(self.paths)} images. Preview: {self.paths[idx].name}"
                )
        else:
            logger.info("Folder selection canceled")

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
        reg = RegParams(
            method=self.reg_method.currentText(),
            model=self.reg_model.currentText(),
            max_iters=self.max_iters.value(),
            eps=self.eps.value(),
            gauss_blur_sigma=self.gauss_sigma.value(),
            use_clahe=self.use_clahe.isChecked(),
            clahe_clip=self.clahe_clip.value(),
            clahe_grid=self.clahe_grid.value(),
            initial_radius=self.init_radius.value(),
            growth_factor=self.growth_factor.value(),
            use_masked_ecc=self.use_masked.isChecked(),
            orb_features=self.orb_features.value(),
            match_ratio=self.match_ratio.value(),
            min_keypoints=self.min_keypoints.value(),
            min_matches=self.min_matches.value(),
            use_ecc_fallback=self.use_ecc_fallback.isChecked(),
        )
        seg = SegParams(
            method=self.seg_method.currentText(),
            invert=self.invert.isChecked(),
            skip_outline=self.skip_outline.isChecked(),
            manual_thresh=self.manual_t.value(),
            adaptive_block=self.adaptive_blk.value(),
            adaptive_C=self.adaptive_C.value(),
            local_block=self.local_blk.value(),
            morph_open_radius=int(self.open_r.text()) if self.open_r.text() else None,
            morph_close_radius=(
                int(self.close_r.text()) if self.close_r.text() else None
            ),
            remove_objects_smaller_px=self.rm_obj.value(),
            remove_holes_smaller_px=self.rm_holes.value(),
            use_clahe=self.use_clahe.isChecked(),
        )
        scale_minmax = (self.scale_min.value(), self.scale_max.value())
        if scale_minmax[1] <= scale_minmax[0]:
            scale_minmax = None
        app = AppParams(
            direction=self.dir_combo.currentText(),
            minutes_between_frames=self.dt_min.value(),
            use_file_timestamps=self.use_ts.isChecked(),
            normalize=self.norm_cb.isChecked(),
            subtract_background=self.bg_sub_cb.isChecked(),
            scale_minmax=scale_minmax,
            use_difference_for_seg=True,
            difference_method=self.diff_method.currentText(),
            show_ref_overlay=self.overlay_ref_cb.isChecked(),
            show_mov_overlay=self.overlay_mov_cb.isChecked(),
            overlay_opacity=self.alpha_slider.value(),
            overlay_mode=self.overlay_mode_combo.currentText(),
            overlay_ref_color=self.ref_color,
            overlay_mov_color=self.mov_color,
            overlay_new_color=self.new_color,
            overlay_lost_color=self.lost_color,
            save_intermediates=self.save_intermediates.isChecked(),
            archive_intermediates=self.archive_intermediates.isChecked(),
            save_masks=self.save_masks_checkbox.isChecked(),
            save_gm_composite=self.save_gm_checkbox.isChecked(),
        )
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
        initial = (
            str(Path(self.app.presets_path) / "preset.json")
            if self.app.presets_path
            else "preset.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", initial, "JSON (*.json)"
        )
        if path:
            self.app.presets_path = str(Path(path).parent)
            save_preset(path, reg, seg, self.app)
            save_settings(self.reg, self.seg, self.app)
            self.status_label.setText(f"Preset saved: {path}")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", self.app.presets_path or "", "JSON (*.json)"
        )
        if not path:
            return
        reg, seg, app = load_preset(path)
        app.presets_path = str(Path(path).parent)
        self.reg = reg
        self.seg = seg
        self.app = app
        # Update UI
        self.reg_method.setCurrentText(reg.method)
        self.reg_model.setCurrentText(reg.model)
        self.max_iters.setValue(reg.max_iters)
        self.eps.setValue(reg.eps)
        self.gauss_sigma.setValue(reg.gauss_blur_sigma)
        self.use_clahe.setChecked(reg.use_clahe or seg.use_clahe)
        self.clahe_clip.setValue(reg.clahe_clip)
        self.clahe_grid.setValue(reg.clahe_grid)
        self.init_radius.setValue(reg.initial_radius)
        self.growth_factor.setValue(reg.growth_factor)
        self.use_masked.setChecked(reg.use_masked_ecc)
        self.orb_features.setValue(reg.orb_features)
        self.match_ratio.setValue(reg.match_ratio)
        self.use_ecc_fallback.setChecked(reg.use_ecc_fallback)
        self.min_keypoints.setValue(reg.min_keypoints)
        self.min_matches.setValue(reg.min_matches)
        self.seg_method.setCurrentText(seg.method)
        self.invert.setChecked(seg.invert)
        self.skip_outline.setChecked(seg.skip_outline)
        self.manual_t.setValue(seg.manual_thresh)
        self.adaptive_blk.setValue(seg.adaptive_block)
        self.adaptive_C.setValue(seg.adaptive_C)
        self.local_blk.setValue(seg.local_block)
        if seg.morph_open_radius is not None:
            self.open_r.setValue(seg.morph_open_radius)
        else:
            self.open_r.lineEdit().clear()
        if seg.morph_close_radius is not None:
            self.close_r.setValue(seg.morph_close_radius)
        else:
            self.close_r.lineEdit().clear()
        self.rm_obj.setValue(seg.remove_objects_smaller_px)
        self.rm_holes.setValue(seg.remove_holes_smaller_px)
        self.dir_combo.setCurrentText(app.direction)
        self.dt_min.setValue(app.minutes_between_frames)
        self.use_ts.setChecked(app.use_file_timestamps)
        self.diff_method.setCurrentText(app.difference_method)
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
        self.save_intermediates.setChecked(app.save_intermediates)
        self.archive_intermediates.setChecked(app.archive_intermediates)
        self.ref_color = tuple(app.overlay_ref_color)
        self.mov_color = tuple(app.overlay_mov_color)
        self.new_color = tuple(app.overlay_new_color)
        self.lost_color = tuple(app.overlay_lost_color)
        self._set_btn_color(self.ref_color_btn, self.ref_color)
        self._set_btn_color(self.mov_color_btn, self.mov_color)
        self._set_btn_color(self.new_color_btn, self.new_color)
        self._set_btn_color(self.lost_color_btn, self.lost_color)
        self._on_overlay_mode_changed(app.overlay_mode)
        self.status_label.setText(f"Preset loaded: {path}")
        self._persist_settings()
        self._on_reg_method_change(self.reg_method.currentText())

    def _on_reg_method_change(self, method: str):
        """Enable or hide method-specific registration controls."""
        is_ecc = method in ("ECC", "ORB+ECC")
        is_orb = method in ("ORB", "ORB+ECC")

        # ECC-specific widgets
        ecc_widgets = [
            (self.max_iters, self.max_iters_label),
            (self.eps, self.eps_label),
            (self.init_radius, self.init_radius_label),
            (self.growth_factor, self.growth_factor_label),
            (self.use_masked, self.use_masked_label),
        ]
        for widget, label in ecc_widgets:
            widget.setEnabled(is_ecc)
            widget.setVisible(is_ecc)
            label.setVisible(is_ecc)

        # ORB-specific widgets
        orb_widgets = [
            (self.orb_features, self.orb_features_label),
            (self.match_ratio, self.match_ratio_label),
            (self.min_keypoints, self.min_keypoints_label),
            (self.min_matches, self.min_matches_label),
            (self.use_ecc_fallback, self.use_ecc_label),
        ]
        for widget, label in orb_widgets:
            widget.setEnabled(is_orb)
            widget.setVisible(is_orb)
            label.setVisible(is_orb)

    def _on_params_changed(self, *args):
        """Debounce rapid param changes and rerun active preview."""
        sender = self.sender()
        if sender is not None:
            try:
                if hasattr(sender, "value"):
                    val = sender.value()
                elif hasattr(sender, "isChecked"):
                    val = sender.isChecked()
                elif hasattr(sender, "currentText"):
                    val = sender.currentText()
                elif hasattr(sender, "text"):
                    val = sender.text()
                else:
                    val = None
            except Exception:
                val = None
            name = sender.objectName() or sender.__class__.__name__
            logger.info("Parameter changed via %s: %s", name, val)
        if (
            sender is not None
            and hasattr(sender, "isEnabled")
            and not sender.isEnabled()
        ):
            return
        if self._current_preview not in ("registration", "segmentation", "difference"):
            return
        self._param_timer.start()

    def _apply_param_change(self):
        logger.info("Applying parameter change for preview: %s", self._current_preview)
        if self._current_preview == "registration":
            self._preview_registration()
        elif self._current_preview == "segmentation":
            self._preview_segmentation()
        elif self._current_preview == "difference":
            self._preview_difference()

    def _refresh_overlay_alpha(self):
        """Blend cached overlays according to slider and checkbox states."""
        alpha = self.alpha_slider.value() / 100.0
        if (
            self._current_preview == "registration"
            and self._reg_ref is not None
            and self._reg_warp is not None
        ):
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
        elif (
            self._current_preview == "segmentation"
            and self._seg_gray is not None
            and self._seg_overlay is not None
        ):
            if self.overlay_ref_cb.isChecked() and self.overlay_mov_cb.isChecked():
                blend = cv2.addWeighted(
                    self._seg_gray, 1 - alpha, self._seg_overlay, alpha, 0
                )
            elif self.overlay_ref_cb.isChecked():
                blend = self._seg_gray
            elif self.overlay_mov_cb.isChecked():
                blend = self._seg_overlay
            else:
                blend = np.zeros_like(self._seg_gray)
            self.view.setImage(blend.transpose(1, 0, 2))
        elif self._current_preview == "difference" and self._diff_img is not None:
            self.view.setImage(self._diff_img.transpose(1, 0, 2))

    def _preview_registration(self):
        # Clear any previous previews so stale images aren't blended
        self._current_preview = None
        self._reg_ref = None
        self._reg_warp = None
        self._seg_gray = None
        self._seg_overlay = None
        self._diff_img = None
        self._diff_gray = None
        # Reset registration flag until this preview completes successfully
        self._registration_done = False
        self.seg_preview_btn.setEnabled(False)
        self.diff_preview_btn.setEnabled(False)

        if len(self.paths) < 2:
            QMessageBox.warning(
                self,
                "Need at least two images",
                "Load at least two images for preview.",
            )
            return
        try:
            reg, _, app = self._persist_settings()
            pair_idx = self.mov_idx_spin.value()
            if pair_idx < 0 or pair_idx > len(self.paths) - 2:
                QMessageBox.warning(
                    self, "Invalid index", "Select a valid frame pair index."
                )
                return
            if app.direction == "first-to-last":
                ref_idx = pair_idx
                mov_idx = pair_idx + 1
            else:
                ref_idx = len(self.paths) - 1 - pair_idx
                mov_idx = ref_idx - 1
            ref_img = imread_gray(
                self.paths[ref_idx],
                normalize=app.normalize,
                scale_minmax=app.scale_minmax,
            )
            mov_img = imread_gray(
                self.paths[mov_idx],
                normalize=app.normalize,
                scale_minmax=app.scale_minmax,
            )
            ref_img = preprocess(
                ref_img,
                reg.gauss_blur_sigma,
                reg.clahe_clip,
                reg.clahe_grid,
                reg.use_clahe,
            )
            mov_img = preprocess(
                mov_img,
                reg.gauss_blur_sigma,
                reg.clahe_clip,
                reg.clahe_grid,
                reg.use_clahe,
            )
            method = reg.method.upper()
            if method == "ORB":
                success, _, warped, _, fb, ref_kp, mov_kp = register_orb(
                    ref_img,
                    mov_img,
                    model=reg.model,
                    orb_features=reg.orb_features,
                    match_ratio=reg.match_ratio,
                    min_keypoints=reg.min_keypoints,
                    min_matches=reg.min_matches,
                    use_ecc_fallback=reg.use_ecc_fallback,
                )
                if fb:
                    self.status_label.setText("ORB fell back to ECC for registration.")
            elif method == "ORB+ECC":
                success, _, warped, _, ref_kp, mov_kp = register_orb_ecc(
                    ref_img,
                    mov_img,
                    model=reg.model,
                    max_iters=reg.max_iters,
                    eps=reg.eps,
                    orb_features=reg.orb_features,
                    match_ratio=reg.match_ratio,
                    min_keypoints=reg.min_keypoints,
                    min_matches=reg.min_matches,
                    use_ecc_fallback=reg.use_ecc_fallback,
                )
            else:
                success, _, warped, _ = register_ecc(
                    ref_img,
                    mov_img,
                    model=reg.model,
                    max_iters=reg.max_iters,
                    eps=reg.eps,
                )
                ref_kp = mov_kp = 0
            if not success:
                raise RuntimeError("Registration failed")
            self.kp_label.setText(f"Keypoints: ref={ref_kp}, mov={mov_kp}")
            self._reg_ref = ref_img
            self._reg_warp = warped
            self._current_preview = "registration"
            # Blend and push the new image to the viewer
            self._refresh_overlay_alpha()
            self.view.setImage(self.view.imageItem.image)
            self.status_label.setText("Preview successful.")
            # Registration succeeded; difference can now be previewed
            self._registration_done = True
            self.diff_preview_btn.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")

    def _preview_difference(self):
        # Clear previous previews that might interfere
        self._current_preview = None
        self._seg_gray = None
        self._seg_overlay = None
        self._diff_img = None
        self._diff_gray = None
        self.seg_preview_btn.setEnabled(False)
        self.diff_preview_btn.setEnabled(False)

        if len(self.paths) < 2:
            QMessageBox.warning(
                self,
                "Need at least two images",
                "Load at least two images for preview.",
            )
            return
        try:
            reg, _, app = self._persist_settings()
            pair_idx = self.mov_idx_spin.value()
            if pair_idx < 0 or pair_idx > len(self.paths) - 2:
                QMessageBox.warning(
                    self, "Invalid index", "Select a valid frame pair index."
                )
                return
            if app.direction == "first-to-last":
                ref_idx = pair_idx
                mov_idx = pair_idx + 1
            else:
                ref_idx = len(self.paths) - 1 - pair_idx
                mov_idx = ref_idx - 1

            if self._reg_ref is None or self._reg_warp is None:
                ref_img = imread_gray(
                    self.paths[ref_idx],
                    normalize=app.normalize,
                    scale_minmax=app.scale_minmax,
                )
                mov_img = imread_gray(
                    self.paths[mov_idx],
                    normalize=app.normalize,
                    scale_minmax=app.scale_minmax,
                )
                ref_img = preprocess(
                    ref_img,
                    reg.gauss_blur_sigma,
                    reg.clahe_clip,
                    reg.clahe_grid,
                    reg.use_clahe,
                )
                mov_img = preprocess(
                    mov_img,
                    reg.gauss_blur_sigma,
                    reg.clahe_clip,
                    reg.clahe_grid,
                    reg.use_clahe,
                )
                method = reg.method.upper()
                if method == "ORB":
                    success, _, warped, _, fb, ref_kp, mov_kp = register_orb(
                        ref_img,
                        mov_img,
                        model=reg.model,
                        orb_features=reg.orb_features,
                        match_ratio=reg.match_ratio,
                        min_keypoints=reg.min_keypoints,
                        min_matches=reg.min_matches,
                        use_ecc_fallback=reg.use_ecc_fallback,
                    )
                    if fb:
                        self.status_label.setText(
                            "ORB fell back to ECC for registration."
                        )
                elif method == "ORB+ECC":
                    success, _, warped, _, ref_kp, mov_kp = register_orb_ecc(
                        ref_img,
                        mov_img,
                        model=reg.model,
                        max_iters=reg.max_iters,
                        eps=reg.eps,
                        orb_features=reg.orb_features,
                        match_ratio=reg.match_ratio,
                        min_keypoints=reg.min_keypoints,
                        min_matches=reg.min_matches,
                        use_ecc_fallback=reg.use_ecc_fallback,
                    )
                else:
                    success, _, warped, _ = register_ecc(
                        ref_img,
                        mov_img,
                        model=reg.model,
                        max_iters=reg.max_iters,
                        eps=reg.eps,
                    )
                    ref_kp = mov_kp = 0
                if not success:
                    raise RuntimeError("Registration failed")
                self.kp_label.setText(f"Keypoints: ref={ref_kp}, mov={mov_kp}")
                self._reg_ref = ref_img
                self._reg_warp = warped

            diff = compute_difference(
                self._reg_ref, self._reg_warp, method=self.diff_method.currentText()
            )
            self._diff_gray = diff
            self._diff_img = cv2.cvtColor(self._diff_gray, cv2.COLOR_GRAY2RGB)
            self._current_preview = "difference"
            self._registration_done = True

            self._refresh_overlay_alpha()
            self.view.setImage(self.view.imageItem.image)
            self.status_label.setText("Difference preview successful.")
            self.seg_preview_btn.setEnabled(True)
            self.diff_preview_btn.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")

    def _preview_segmentation(self):
        # Clear any previous previews so stale images aren't blended
        self._current_preview = None
        self._seg_gray = None
        self._seg_overlay = None

        if self._diff_gray is None:
            QMessageBox.warning(
                self, "Run difference preview", "Run the difference preview first."
            )
            return
        try:
            _, seg, _ = self._persist_settings()
            gray = self._diff_gray
            # Mirror pipeline input handling: pass the raw frame to ``segment`` and
            # only normalize when necessary. ``compute_difference`` and
            # ``imread_gray`` already yield ``uint8`` images, so avoid double
            # scaling unless a non-uint8 array arrives here.
            if gray.dtype != np.uint8:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                )
            bw = segment(
                gray,
                method=seg.method,
                invert=seg.invert,
                skip_outline=seg.skip_outline,
                use_diff=True,
                manual_thresh=seg.manual_thresh,
                adaptive_block=seg.adaptive_block,
                adaptive_C=seg.adaptive_C,
                local_block=seg.local_block,
                morph_open_radius=seg.morph_open_radius,
                morph_close_radius=seg.morph_close_radius,
                remove_objects_smaller_px=seg.remove_objects_smaller_px,
                remove_holes_smaller_px=seg.remove_holes_smaller_px,
                use_clahe=seg.use_clahe,
            )

            self._seg_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self._seg_overlay = cv2.cvtColor(
                overlay_outline(gray, bw), cv2.COLOR_BGR2RGB
            )
            self._current_preview = "segmentation"
            self._registration_done = True

            # Blend and push the new image to the viewer
            self._refresh_overlay_alpha()
            self.view.setImage(self.view.imageItem.image)
            self.status_label.setText("Segmentation preview successful.")
            self.diff_preview_btn.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")

    def _run_pipeline(self):
        if not self.paths:
            QMessageBox.warning(self, "No images", "Choose an image folder first.")
            return

        reg, seg, app = self._persist_settings()
        # Guard against invalid direction values
        if app.direction not in ("first-to-last", "last-to-first"):
            QMessageBox.critical(
                self,
                "Invalid Direction",
                f"Analysis direction must be 'first-to-last' or 'last-to-first'. Got: {app.direction}",
            )
            return

        # Warn about small initial radius
        try:
            img0 = imread_gray(self.paths[0], normalize=False)
            h, w = img0.shape[:2]
            r = reg.initial_radius
            if r > 0:
                cx, cy = w // 2, h // 2
                x0 = max(cx - r, 0)
                y0 = max(cy - r, 0)
                x1 = min(cx + r, w)
                y1 = min(cy + r, h)
                mask_area = (x1 - x0) * (y1 - y0)
                if mask_area < 0.25 * h * w:
                    msg = QMessageBox(
                        QMessageBox.Icon.Warning,
                        "Small Initial Radius",
                        (
                            f"Initial radius {r} covers only {mask_area/(h*w)*100:.1f}% of the frame.\n"
                            "Use full frame instead?"
                        ),
                        parent=self,
                    )
                    msg.setStandardButtons(
                        QMessageBox.StandardButton.Yes
                        | QMessageBox.StandardButton.Cancel
                    )
                    msg.button(QMessageBox.StandardButton.Yes).setText("Full Frame")
                    if msg.exec() == QMessageBox.StandardButton.Yes:
                        self.init_radius.setValue(0)
                        reg, seg, app = self._persist_settings()
                    else:
                        return
        except Exception as e:
            logger.warning("Could not validate initial radius: %s", e)

        logger.info("Run Analysis button clicked with direction=%s", app.direction)

        # Build slim dicts for worker. seg_cfg mirrors the segmentation preview
        # parameters and is forwarded to PipelineWorker unchanged.
        reg_cfg = dict(
            method=reg.method,
            model=reg.model,
            max_iters=reg.max_iters,
            eps=reg.eps,
            use_masked_ecc=reg.use_masked_ecc,
            gauss_blur_sigma=reg.gauss_blur_sigma,
            use_clahe=reg.use_clahe,
            clahe_clip=reg.clahe_clip,
            clahe_grid=reg.clahe_grid,
            initial_radius=reg.initial_radius,
            growth_factor=reg.growth_factor,
            orb_features=reg.orb_features,
            match_ratio=reg.match_ratio,
            min_keypoints=reg.min_keypoints,
            min_matches=reg.min_matches,
            use_ecc_fallback=reg.use_ecc_fallback,
        )
        seg_cfg = dict(
            method=seg.method,
            invert=seg.invert,
            skip_outline=seg.skip_outline,
            manual_thresh=seg.manual_thresh,
            adaptive_block=seg.adaptive_block,
            adaptive_C=seg.adaptive_C,
            local_block=seg.local_block,
            morph_open_radius=seg.morph_open_radius,
            morph_close_radius=seg.morph_close_radius,
            remove_objects_smaller_px=seg.remove_objects_smaller_px,
            remove_holes_smaller_px=seg.remove_holes_smaller_px,
        )
        app_cfg = dict(
            direction=app.direction,
            use_difference_for_seg=app.use_difference_for_seg,
            save_intermediates=app.save_intermediates,
            archive_intermediates=app.archive_intermediates,
            save_masks=self.save_masks_checkbox.isChecked(),
            save_gm_composite=self.save_gm_checkbox.isChecked(),
            difference_method=app.difference_method,
            normalize=app.normalize,
            subtract_background=app.subtract_background,
            scale_minmax=app.scale_minmax,
        )

        out_dir = Path(self.folder_edit.text()) / "_processed_pyqt"
        self.thread = QThread()
        self.worker = PipelineWorker(self.paths, reg_cfg, seg_cfg, app_cfg, out_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.started.connect(lambda: logger.info("Pipeline thread started"))
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.thread.start()
        self.status_label.setText("Processing…")

    def _on_done(self, out_dir: str):
        logger.info("Pipeline thread finished successfully: %s", out_dir)
        self.status_label.setText(f"Done. Outputs: {out_dir}")
        self.thread.quit()
        self.thread.wait()

    def _on_failed(self, err: str):
        logger.error("Pipeline thread failed: %s", err)
        QMessageBox.critical(
            self, "Processing Error", f"{err}\nNo summary file was written."
        )
        self.status_label.setText(f"Failed: {err}")
        self.thread.quit()
        self.thread.wait()

    def closeEvent(self, event):
        """Persist current settings when the window is closed."""
        self._persist_settings()
        super().closeEvent(event)
