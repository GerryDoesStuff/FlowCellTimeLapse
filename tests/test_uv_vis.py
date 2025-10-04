import numpy as np
import pytest

from app.core.uv_vis import load_uvvis_dataset, compute_uvvis_metrics
from app.models.config import UvVisParams


def _write_csv(path, rows):
    lines = ["wavelength,absorbance\n"]
    for wavelength, absorbance in rows:
        lines.append(f"{wavelength},{absorbance}\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_load_uvvis_dataset_with_references(tmp_path):
    sample = tmp_path / "sample.csv"
    blank = tmp_path / "blank.csv"
    dark = tmp_path / "dark.csv"

    _write_csv(
        sample,
        [
            (200, 0.10),
            (210, 0.50),
            (220, 0.90),
            (230, 0.40),
            (240, 0.20),
        ],
    )
    _write_csv(blank, [(200, 0.05), (210, 0.05), (220, 0.05), (230, 0.05), (240, 0.05)])
    _write_csv(dark, [(200, 0.01), (210, 0.01), (220, 0.01), (230, 0.01), (240, 0.01)])

    params = UvVisParams(
        data_files=[str(sample)],
        blank_reference=str(blank),
        dark_reference=str(dark),
        apply_baseline_correction=True,
        baseline_poly_order=1,
        apply_smoothing=True,
        smoothing_window=3,
        ratio_wavelengths=[210.0, 220.0],
        peak_prominence=0.05,
        top_n_peaks=2,
    )

    dataset = load_uvvis_dataset(params)
    assert len(dataset.spectra) == 1
    assert dataset.blank is not None
    assert dataset.dark is not None

    spectrum = dataset.spectra[0]
    np.testing.assert_allclose(
        spectrum.raw_absorbance,
        np.array([0.10, 0.50, 0.90, 0.40, 0.20]),
    )
    assert spectrum.absorbance[2] > spectrum.absorbance[1]
    assert spectrum.absorbance[2] > spectrum.absorbance[3]

    metrics = compute_uvvis_metrics(dataset, params)
    spec_metrics = metrics["spectra"][0]
    assert spec_metrics["peaks"][0]["wavelength"] == pytest.approx(220, abs=1.0)
    assert "area" in spec_metrics
    assert "ratio" in spec_metrics["ratios"]


def test_directory_loading_excludes_references(tmp_path):
    sample = tmp_path / "sample.csv"
    blank = tmp_path / "blank.csv"
    dark = tmp_path / "dark.csv"

    _write_csv(sample, [(200, 0.2), (210, 0.3), (220, 0.4)])
    _write_csv(blank, [(200, 0.1), (210, 0.1), (220, 0.1)])
    _write_csv(dark, [(200, 0.05), (210, 0.05), (220, 0.05)])

    params = UvVisParams(
        data_directory=str(tmp_path),
        blank_reference=str(blank),
        dark_reference=str(dark),
        apply_baseline_correction=False,
        apply_smoothing=False,
    )

    dataset = load_uvvis_dataset(params)
    assert len(dataset.spectra) == 1
    np.testing.assert_allclose(
        dataset.spectra[0].raw_absorbance,
        np.array([0.2, 0.3, 0.4]),
    )
