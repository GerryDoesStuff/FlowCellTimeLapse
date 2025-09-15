from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any
import json
from PyQt6.QtCore import QSettings

RegistrationModel = Literal["translation","euclidean","affine","homography"]
RegMethod = Literal["ECC", "ORB", "ORB+ECC"]
SegMethod = Literal[
    "otsu",
    "multi_otsu",
    "li",
    "yen",
    "adaptive",
    "local",
    "manual",
]

@dataclass
class RegParams:
    method: RegMethod = "ECC"  # "ECC", "ORB", or "ORB+ECC"
    model: RegistrationModel = "affine"
    max_iters: int = 1000
    eps: float = 1e-6
    gauss_blur_sigma: float = 1.0
    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    initial_radius: int = 20
    growth_factor: float = 1.0
    use_masked_ecc: bool = True
    orb_features: int = 4000
    match_ratio: float = 0.75
    min_keypoints: int = 8
    min_matches: int = 8
    use_ecc_fallback: bool = True

@dataclass
class SegParams:
    method: SegMethod = "otsu"
    manual_thresh: int = 128
    adaptive_block: int = 51
    adaptive_C: int = 5
    local_block: int = 51
    remove_holes_smaller_px: int = 0
    remove_objects_smaller_px: int = 0
    morph_open_radius: int | None = None
    morph_close_radius: int | None = None
    invert: bool = True  # cells darker
    skip_outline: bool = False
    use_clahe: bool = False

@dataclass
class AppParams:
    px_size_um: float = 1.0
    minutes_between_frames: float = 1.0  # default; can be overridden by timestamps
    direction: str = "last-to-first"  # "last-to-first" | "first-to-last"
    show_ref_overlay: bool = True
    show_mov_overlay: bool = True
    overlay_opacity: int = 50  # 0-100 weight of moving frame in registration overlay
    overlay_mode: str = "magenta-green"
    overlay_ref_color: tuple[int, int, int] = (0, 255, 0)
    overlay_mov_color: tuple[int, int, int] = (255, 0, 255)
    overlay_new_color: tuple[int, int, int] = (0, 255, 0)
    overlay_lost_color: tuple[int, int, int] = (0, 0, 255)
    gm_opacity: int = 50  # 0-100 weight of current frame in green/magenta composites
    save_jpg_quality: int = 95
    save_diagnostics: bool = True  # save optional diagnostic outputs
    archive_outputs: bool = False  # zip and remove image outputs
    use_difference_for_seg: bool = False  # diff masks saved regardless
    difference_method: str = "abs"
    gm_thresh_method: str = "otsu"  # "otsu" | "percentile"
    gm_thresh_percentile: float = 99.0
    gm_close_kernel: int = 3  # closing kernel size; 0 disables
    gm_dilate_kernel: int = 0  # dilation kernel size; 0 disables
    gm_saturation: float = 1.0  # scale factor for a-channel before thresholding
    show_diff_overlay: bool = True  # draw diff/green and diff/magenta outlines
    use_file_timestamps: bool = True
    normalize: bool = True
    subtract_background: bool = False
    scale_minmax: Optional[tuple[int, int]] = None
    presets_path: Optional[str] = None
    last_folder: str | None = None
    process_subdirs: bool = False

def save_preset(path: str, reg: RegParams, seg: SegParams, app: AppParams) -> None:
    data = {"reg": asdict(reg), "seg": asdict(seg), "app": asdict(app)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_preset(path: str) -> tuple[RegParams, SegParams, AppParams]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    app_data: Dict[str, Any] = data["app"]
    app_data.setdefault("gm_saturation", 1.0)
    app_data.setdefault("gm_opacity", app_data.get("overlay_opacity", 50))
    app_data.setdefault("save_diagnostics", True)
    app_data.setdefault("show_diff_overlay", True)
    app_data.setdefault("archive_outputs", False)
    app_data.setdefault("process_subdirs", False)
    for key in [
        "save_png",
        "save_intermediates",
        "archive_intermediates",
        "save_masks",
        "save_gm_composite",
    ]:
        app_data.pop(key, None)
    return RegParams(**data["reg"]), SegParams(**data["seg"]), AppParams(**app_data)

def save_settings(reg: RegParams, seg: SegParams, app: AppParams) -> None:
    s = QSettings("YeastLab", "FlowcellPyQt")
    s.setValue("reg", json.dumps(asdict(reg)))
    s.setValue("seg", json.dumps(asdict(seg)))
    s.setValue("app", json.dumps(asdict(app)))
    s.sync()

def load_settings() -> tuple[RegParams, SegParams, AppParams]:
    s = QSettings("YeastLab", "FlowcellPyQt")
    reg = s.value("reg"); seg = s.value("seg"); app = s.value("app")
    def parse(v, cls, default):
        if v is None:
            return default
        try:
            data = json.loads(v)
            if cls is AppParams:
                data.setdefault("gm_saturation", 1.0)
                data.setdefault("gm_opacity", data.get("overlay_opacity", 50))
                data.setdefault("save_diagnostics", True)
                data.setdefault("show_diff_overlay", True)
                data.setdefault("archive_outputs", False)
                data.setdefault("process_subdirs", False)
                for key in [
                    "save_png",
                    "save_intermediates",
                    "archive_intermediates",
                    "save_masks",
                    "save_gm_composite",
                ]:
                    data.pop(key, None)
            return cls(**data)
        except Exception:
            return default
    return (
        parse(reg, RegParams, RegParams()),
        parse(seg, SegParams, SegParams()),
        parse(app, AppParams, AppParams()),
    )
