from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mne
from nibabel.freesurfer.io import read_annot
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import opm_source_toolbox
from opm_source_toolbox.atlas_fetch import fetch_schaefer_annotations, import_annotation_pair
from opm_source_toolbox.core import DEFAULT_ATLAS_PARC, resolve_atlas_annotation_spec
from opm_source_toolbox.roi_surface_render import render_roi_value_map_to_surface
from opm_source_toolbox.sensor_to_roi import (
    RoiProjectionResult,
    SensorMatrixSpec,
    SourceProjectionConfig,
    _load_sensor_item,
    export_manifest_to_rois,
)


def test_top_level_public_api_exports_expected_symbols() -> None:
    expected = {
        "AlignmentQcResult",
        "AtlasFetchResult",
        "DEFAULT_ATLAS_PARC",
        "RoiProjectionResult",
        "SensorMatrixSpec",
        "SourceProjectionConfig",
        "SubjectProjectionSpec",
        "SurfaceRenderConfig",
        "default_atlas_subjects_dir",
        "export_manifest_to_rois",
        "export_subject_sensor_matrices",
        "fetch_atlas",
        "fetch_atlas_to_path",
        "fetch_schaefer_annotations",
        "import_annotation_pair",
        "load_roi_value_map_csv",
        "load_subject_specs_from_manifest",
        "render_alignment_qc_bundle",
        "render_alignment_screenshot",
        "render_roi_value_map_to_surface",
        "render_roi_vector_to_surface",
    }

    assert expected.issubset(set(opm_source_toolbox.__all__))
    resolved = {name: getattr(opm_source_toolbox, name) for name in expected}
    assert resolved["DEFAULT_ATLAS_PARC"] == DEFAULT_ATLAS_PARC
    assert resolved["SourceProjectionConfig"] is SourceProjectionConfig
    assert resolved["SensorMatrixSpec"] is SensorMatrixSpec
    assert resolved["RoiProjectionResult"] is RoiProjectionResult


def _make_info() -> mne.Info:
    return mne.create_info(["MEG001", "MEG002"], sfreq=100.0, ch_types=["mag", "mag"])


def _write_dummy_trans(path: Path) -> Path:
    trans = mne.transforms.Transform("head", "mri")
    mne.write_trans(str(path), trans, overwrite=True)
    return path


def _write_raw_fif(path: Path) -> np.ndarray:
    data = np.array(
        [
            np.linspace(0.0, 0.9, 10),
            np.linspace(1.0, 1.9, 10),
        ]
    )
    raw = mne.io.RawArray(data, _make_info(), verbose=False)
    raw.save(str(path), overwrite=True)
    return data


def _write_epochs_fif(path: Path) -> np.ndarray:
    data = np.stack(
        [
            np.full((2, 10), 1.0),
            np.full((2, 10), 3.0),
        ],
        axis=0,
    )
    epochs = mne.EpochsArray(data, _make_info(), tmin=-0.05, verbose=False)
    epochs.save(str(path), overwrite=True)
    return data


def _write_evoked_fif(path: Path) -> np.ndarray:
    left = mne.EvokedArray(
        np.full((2, 10), 7.0),
        _make_info(),
        tmin=0.0,
        comment="left",
        verbose=False,
    )
    right = mne.EvokedArray(
        np.full((2, 10), 9.0),
        _make_info(),
        tmin=0.0,
        comment="right",
        verbose=False,
    )
    mne.write_evokeds(str(path), [left, right], overwrite=True)
    return left.data


@pytest.mark.parametrize(
    ("kind", "builder", "spec_kwargs", "expected"),
    [
        (
            "raw",
            _write_raw_fif,
            {"fif_kind": "raw"},
            np.array(
                [
                    np.linspace(0.0, 0.9, 10),
                    np.linspace(1.0, 1.9, 10),
                ]
            ),
        ),
        (
            "epochs",
            _write_epochs_fif,
            {"fif_kind": "epochs", "epochs_average": True},
            np.full((2, 10), 2.0),
        ),
        (
            "evoked",
            _write_evoked_fif,
            {"fif_kind": "evoked", "evoked_comment": "left"},
            np.full((2, 10), 7.0),
        ),
    ],
)
def test_load_sensor_item_supports_direct_fif_kinds(
    tmp_path: Path,
    kind: str,
    builder,
    spec_kwargs: dict[str, object],
    expected: np.ndarray,
) -> None:
    trans_path = _write_dummy_trans(tmp_path / "sample-trans.fif")
    suffix = {"raw": "_raw.fif", "epochs": "-epo.fif", "evoked": "-ave.fif"}[kind]
    fif_path = tmp_path / f"sample_{kind}{suffix}"
    builder(fif_path)

    loaded = _load_sensor_item(
        SensorMatrixSpec(
            name=f"{kind}_item",
            fif_path=str(fif_path),
            trans_path=str(trans_path),
            **spec_kwargs,
        )
    )

    assert loaded.input_kind == kind
    assert loaded.ch_names == ["MEG001", "MEG002"]
    assert loaded.data.shape == (2, 10)
    assert loaded.sfreq_hz == pytest.approx(100.0)
    assert np.allclose(loaded.data, expected)


def test_export_manifest_to_rois_from_raw_fif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject_dir = tmp_path / "R9999"
    (subject_dir / "FS").mkdir(parents=True)
    fif_path = tmp_path / "preprocessed_R9999_Run01_raw.fif"
    trans_path = _write_dummy_trans(tmp_path / "R9999_Run01_trans.fif")
    _write_raw_fif(fif_path)

    manifest = {
        "subjects": [
            {
                "subject": "R9999",
                "subject_dir": str(subject_dir),
                "fs_subject": "FS",
                "items": [
                    {
                        "name": "run01_window",
                        "fif_path": str(fif_path),
                        "fif_kind": "raw",
                        "trans_path": str(trans_path),
                        "tmin_s": 0.02,
                        "tmax_s": 0.05,
                        "metadata": {"condition": "rest"},
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr("opm_source_toolbox.sensor_to_roi.setup_subject_source_space", lambda **_: "src")
    monkeypatch.setattr(
        "opm_source_toolbox.sensor_to_roi.load_atlas_labels",
        lambda **_: [SimpleNamespace(name="ROI_A"), SimpleNamespace(name="ROI_B")],
    )

    def _fake_project(**kwargs):
        loaded_item = kwargs["loaded_item"]
        data = np.vstack([loaded_item.data[0], loaded_item.data[1]])
        return RoiProjectionResult(
            item_name=loaded_item.spec.name,
            roi_names=["ROI_A", "ROI_B"],
            data=data,
            source_file=str(loaded_item.spec.source_file),
            geometry_info_file=str(loaded_item.geometry_info_file),
            trans_path=str(loaded_item.spec.trans_path),
            n_input_channels=len(loaded_item.ch_names),
            n_used_channels=len(loaded_item.ch_names),
            metadata=dict(loaded_item.spec.metadata),
        )

    monkeypatch.setattr("opm_source_toolbox.sensor_to_roi.project_sensor_item_to_atlas_rois", _fake_project)

    out_paths = export_manifest_to_rois(
        manifest_path=str(manifest_path),
        out_root=str(tmp_path / "exports"),
        config=SourceProjectionConfig(),
    )

    assert len(out_paths) == 1
    subject_out = Path(out_paths[0])
    out_csv = subject_out / "run01_window.csv"
    meta_path = subject_out / "metadata.json"
    assert out_csv.exists()
    assert meta_path.exists()

    exported = pd.read_csv(out_csv)
    assert list(exported["roi_name"]) == ["ROI_A", "ROI_B"]
    assert [col for col in exported.columns if col.startswith("t")] == ["t000", "t001", "t002", "t003"]

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["subject"] == "R9999"
    assert metadata["n_items"] == 1
    assert metadata["n_rois"] == 2
    assert metadata["items"][0]["input_kind"] == "raw"
    assert metadata["items"][0]["metadata"] == {"condition": "rest"}
    assert metadata["items"][0]["sfreq_hz"] == pytest.approx(100.0)
    assert metadata["items"][0]["time_start_s"] == pytest.approx(0.02)
    assert metadata["items"][0]["n_timepoints"] == 4


def test_packaged_schaefer_annotations_resolve_from_generic_package(tmp_path: Path) -> None:
    atlas_subjects_dir = tmp_path / "subjects"
    (atlas_subjects_dir / "fsaverage").mkdir(parents=True)
    subject_dir = tmp_path / "subject"
    (subject_dir / "FS" / "label").mkdir(parents=True)

    spec = resolve_atlas_annotation_spec(
        subject_dir=str(subject_dir),
        fs_subject="FS",
        atlas_parc=DEFAULT_ATLAS_PARC,
        atlas_subject="fsaverage",
        atlas_subjects_dir=str(atlas_subjects_dir),
    )

    assert spec["uses_packaged_annotations"] is True
    assert all("opm_source_toolbox/schaefer" in path for path in spec["atlas_paths"])


def test_fetch_schaefer_annotations_installs_into_subjects_dir(tmp_path: Path) -> None:
    result = fetch_schaefer_annotations(atlas_subjects_dir=str(tmp_path / "subjects"))

    assert result.atlas_name == "schaefer"
    assert result.atlas_subject in {"fsaverage", "fsaverage4"}
    assert Path(result.label_dir).exists()
    assert all(Path(path).exists() for path in result.atlas_paths)


def test_import_annotation_pair_copies_custom_annotations(tmp_path: Path) -> None:
    package_root = Path(opm_source_toolbox.__file__).resolve().parent
    lh_src = package_root / "schaefer" / f"lh.{DEFAULT_ATLAS_PARC}.annot"
    rh_src = package_root / "schaefer" / f"rh.{DEFAULT_ATLAS_PARC}.annot"

    result = import_annotation_pair(
        atlas_name="custom",
        atlas_parc="CustomAtlas",
        lh_annot_path=str(lh_src),
        rh_annot_path=str(rh_src),
        atlas_subjects_dir=str(tmp_path / "subjects"),
    )

    assert result.atlas_name == "custom"
    assert result.atlas_parc == "CustomAtlas"
    assert all(Path(path).exists() for path in result.atlas_paths)
    assert Path(result.atlas_paths[0]).name == "lh.CustomAtlas.annot"
    assert Path(result.atlas_paths[1]).name == "rh.CustomAtlas.annot"


def test_render_roi_value_map_to_surface_builds_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = Path(opm_source_toolbox.__file__).resolve().parent
    lh_annot = package_root / "schaefer" / f"lh.{DEFAULT_ATLAS_PARC}.annot"
    rh_annot = package_root / "schaefer" / f"rh.{DEFAULT_ATLAS_PARC}.annot"

    _, _, lh_names = read_annot(str(lh_annot))
    _, _, rh_names = read_annot(str(rh_annot))
    lh_roi = next(name.decode("utf-8") for name in lh_names if b"unknown" not in name.lower())
    rh_roi = next(name.decode("utf-8") for name in rh_names if b"unknown" not in name.lower())

    monkeypatch.setattr(
        "opm_source_toolbox.roi_surface_render._load_atlas_surface_spec",
        lambda config: {
            "atlas_subject": "fsaverage",
            "subjects_dir": str(tmp_path),
            "surf_root": str(tmp_path / "surf"),
            "annot_paths": [str(lh_annot), str(rh_annot)],
        },
    )
    monkeypatch.setattr(
        "opm_source_toolbox.roi_surface_render._load_binarized_bg_map",
        lambda *args, **kwargs: np.zeros(10, dtype=float),
    )

    def _fake_tile_renderer(**kwargs) -> None:
        Image.new("RGB", (160, 120), "white").save(kwargs["out_path"])

    monkeypatch.setattr(
        "opm_source_toolbox.roi_surface_render._render_tile_png",
        _fake_tile_renderer,
    )

    result = render_roi_value_map_to_surface(
        value_by_name={lh_roi: 1.0, rh_roi: 2.0},
        out_dir=str(tmp_path / "renders"),
        stem="demo",
        title="Demo",
    )

    assert Path(result["montage_path"]).exists()
    assert len(result["tile_paths"]) == 4
    assert all(Path(path).exists() for path in result["tile_paths"])
    assert result["atlas_subject"] == "fsaverage"
