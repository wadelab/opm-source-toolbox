from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import mne
from nibabel.freesurfer.io import read_annot
import numpy as np
import pandas as pd
import pytest

import opm_source_toolbox
from opm_source_toolbox.atlas_fetch import fetch_schaefer_annotations, import_annotation_pair
from opm_source_toolbox.cli import export_sensor_data_to_source_rois as export_cli
from opm_source_toolbox.core import DEFAULT_ATLAS_PARC, resolve_atlas_annotation_spec
from opm_source_toolbox.sensor_to_roi import (
    RoiProjectionResult,
    SensorMatrixSpec,
    SourceProjectionConfig,
    _load_sensor_item,
    export_manifest_to_rois,
    load_subject_specs_from_manifest,
)
from opm_source_toolbox.workflow import (
    build_bem,
    DEFAULT_SAMPLE_FILES,
    DEFAULT_SAMPLE_FS_URL,
    DEFAULT_YORK_FS_SAMPLE_URL,
    DEFAULT_YORK_SAMPLE_FILES,
    EventDetectionResult,
    detect_primary_events,
    download_file,
    extract_zip_once,
    find_freesurfer_subject_dir,
    prepare_sample_dataset,
    prepare_york_sample_dataset,
    SampleDataset,
    YorkSampleDataset,
)


def test_top_level_public_api_exports_expected_symbols() -> None:
    core_expected = {
        "AtlasFetchResult",
        "DEFAULT_ATLAS_PARC",
        "build_bem",
        "DEFAULT_SAMPLE_FILES",
        "DEFAULT_SAMPLE_FS_URL",
        "DEFAULT_YORK_FS_SAMPLE_URL",
        "DEFAULT_YORK_SAMPLE_FILES",
        "detect_primary_events",
        "download_file",
        "EventDetectionResult",
        "RoiProjectionResult",
        "SensorMatrixSpec",
        "SourceProjectionConfig",
        "SubjectProjectionSpec",
        "default_atlas_subjects_dir",
        "export_manifest_to_rois",
        "export_subject_sensor_matrices",
        "extract_zip_once",
        "fetch_atlas",
        "fetch_atlas_to_path",
        "fetch_schaefer_annotations",
        "find_freesurfer_subject_dir",
        "import_annotation_pair",
        "load_subject_specs_from_manifest",
        "prepare_sample_dataset",
        "prepare_york_sample_dataset",
        "SampleDataset",
        "YorkSampleDataset",
    }
    alignment_expected = {
        "AlignmentQcResult",
        "render_alignment_qc_bundle",
        "render_alignment_screenshot",
    }
    surface_expected = {
        "SurfaceRenderConfig",
        "load_roi_value_map_csv",
        "render_roi_value_map_to_surface",
        "render_roi_vector_to_surface",
    }

    assert core_expected.union(alignment_expected).union(surface_expected).issubset(
        set(opm_source_toolbox.__all__)
    )
    resolved = {name: getattr(opm_source_toolbox, name) for name in core_expected}
    assert resolved["DEFAULT_ATLAS_PARC"] == DEFAULT_ATLAS_PARC
    assert resolved["SourceProjectionConfig"] is SourceProjectionConfig
    assert resolved["SensorMatrixSpec"] is SensorMatrixSpec
    assert resolved["RoiProjectionResult"] is RoiProjectionResult
    assert resolved["EventDetectionResult"] is EventDetectionResult
    assert resolved["SampleDataset"] is SampleDataset
    assert resolved["YorkSampleDataset"] is YorkSampleDataset

    try:
        import nilearn  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        surface_resolved = {name: getattr(opm_source_toolbox, name) for name in surface_expected}
        assert set(surface_resolved) == surface_expected

    try:
        import PIL  # noqa: F401
        import pyvista  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        alignment_resolved = {
            name: getattr(opm_source_toolbox, name) for name in alignment_expected
        }
        assert set(alignment_resolved) == alignment_expected


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


def _write_sensor_matrix_csv(path: Path) -> np.ndarray:
    data = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2, 1.3],
        ]
    )
    pd.DataFrame(
        {
            "ch_name": ["MEG001", "MEG002"],
            "t000": data[:, 0],
            "t001": data[:, 1],
            "t002": data[:, 2],
            "t003": data[:, 3],
        }
    ).to_csv(path, index=False)
    return data


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


def test_load_sensor_item_supports_matrix_csv_with_explicit_sfreq(tmp_path: Path) -> None:
    trans_path = _write_dummy_trans(tmp_path / "sample-trans.fif")
    source_fif = tmp_path / "sample_raw.fif"
    matrix_path = tmp_path / "sample_matrix.csv"
    _write_raw_fif(source_fif)
    expected = _write_sensor_matrix_csv(matrix_path)

    loaded = _load_sensor_item(
        SensorMatrixSpec(
            name="csv_item",
            matrix_path=str(matrix_path),
            sfreq_hz=200.0,
            time_start_s=-0.05,
            source_file=str(source_fif),
            trans_path=str(trans_path),
        )
    )

    assert loaded.input_kind == "csv"
    assert loaded.ch_names == ["MEG001", "MEG002"]
    assert loaded.sfreq_hz == pytest.approx(200.0)
    assert loaded.time_start_s == pytest.approx(-0.05)
    assert np.allclose(loaded.data, expected)


def test_download_file_supports_file_urls(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"sample-bytes")

    destination = tmp_path / "downloads" / "copy.bin"
    out_path = download_file(source.as_uri(), destination)

    assert out_path == destination
    assert destination.read_bytes() == b"sample-bytes"


def test_extract_zip_once_uses_marker_file(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/example.txt", "hello")

    extracted = extract_zip_once(archive_path, tmp_path / "unzipped")
    assert (extracted / "nested" / "example.txt").read_text(encoding="utf-8") == "hello"

    (extracted / "nested" / "example.txt").write_text("changed", encoding="utf-8")
    extracted_again = extract_zip_once(archive_path, extracted)

    assert extracted_again == extracted
    assert (extracted / "nested" / "example.txt").read_text(encoding="utf-8") == "changed"


def test_find_freesurfer_subject_dir_finds_subject_root(tmp_path: Path) -> None:
    subject_dir = tmp_path / "subjects_download" / "FS"
    (subject_dir / "mri").mkdir(parents=True)
    (subject_dir / "surf").mkdir(parents=True)
    (subject_dir / "mri" / "orig.mgz").write_text("x", encoding="utf-8")
    (subject_dir / "surf" / "lh.white").write_text("x", encoding="utf-8")

    assert find_freesurfer_subject_dir(tmp_path / "subjects_download") == subject_dir


def test_detect_primary_events_prefers_annotations() -> None:
    raw = mne.io.RawArray(np.zeros((1, 20)), mne.create_info(["MEG001"], 100.0, ["mag"]))
    raw.set_annotations(
        mne.Annotations(
            onset=[0.01, 0.05, 0.09],
            duration=[0.0, 0.0, 0.0],
            description=["visual", "visual", "bad_motion"],
        )
    )

    result = detect_primary_events(raw)

    assert isinstance(result, EventDetectionResult)
    assert result.source == "annotations"
    assert result.event_id == {"visual": 1}
    assert result.events.shape[0] == 2


def test_detect_primary_events_falls_back_to_stim_channel() -> None:
    data = np.zeros((2, 40))
    data[1, [5, 10, 30]] = [1, 1, 2]
    raw = mne.io.RawArray(
        data,
        mne.create_info(["MEG001", "STI 014"], 100.0, ["mag", "stim"]),
        verbose=False,
    )

    result = detect_primary_events(raw)

    assert result.source == "STI 014"
    assert result.event_id == {"visual": 1}
    assert result.events.shape[0] == 3


def test_prepare_york_sample_dataset_downloads_and_resolves_paths(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    inhelmet = source_dir / "Rxxxx_InHelmet.ply"
    outside = source_dir / "Rxxxx_01_Outside.ply"
    mri_scalp = source_dir / "mriscalp.stl"
    raw_fif_src = source_dir / "VEP_DS-raw.fif"
    inhelmet.write_text("inside", encoding="utf-8")
    outside.write_text("outside", encoding="utf-8")
    mri_scalp.write_text("scalp", encoding="utf-8")
    raw_fif_src.write_text("raw", encoding="utf-8")

    fs_zip = source_dir / "FS_Sample.zip"
    with zipfile.ZipFile(fs_zip, "w") as archive:
        archive.writestr("FS/mri/orig.mgz", "orig")
        archive.writestr("FS/surf/lh.white", "white")

    sample_files = {
        "Rxxxx_InHelmet.ply": inhelmet.as_uri(),
        "Rxxxx_01_Outside.ply": outside.as_uri(),
        "FS/surf/mriscalp.stl": mri_scalp.as_uri(),
        "VEP_DS-raw.fif": raw_fif_src.as_uri(),
    }

    result = prepare_york_sample_dataset(
        work_dir=tmp_path / "work",
        sample_files=sample_files,
        fs_sample_url=fs_zip.as_uri(),
    )

    assert isinstance(result, YorkSampleDataset)
    assert result.subject_name == "FS"
    assert result.subject_dir == result.subjects_download_dir / "FS"
    assert result.raw_fif.exists()
    assert result.aligned_fif.exists()
    assert result.inside_mesh.exists()
    assert result.outside_mesh.exists()
    assert result.mri_scalp.exists()


def test_prepare_sample_dataset_is_alias_of_packaged_helper(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    inhelmet = source_dir / "Rxxxx_InHelmet.ply"
    outside = source_dir / "Rxxxx_01_Outside.ply"
    mri_scalp = source_dir / "mriscalp.stl"
    raw_fif_src = source_dir / "VEP_DS-raw.fif"
    inhelmet.write_text("inside", encoding="utf-8")
    outside.write_text("outside", encoding="utf-8")
    mri_scalp.write_text("scalp", encoding="utf-8")
    raw_fif_src.write_text("raw", encoding="utf-8")

    fs_zip = source_dir / "FS_Sample.zip"
    with zipfile.ZipFile(fs_zip, "w") as archive:
        archive.writestr("FS/mri/orig.mgz", "orig")
        archive.writestr("FS/surf/lh.white", "white")

    result = prepare_sample_dataset(
        work_dir=tmp_path / "work",
        sample_files={
            "Rxxxx_InHelmet.ply": inhelmet.as_uri(),
            "Rxxxx_01_Outside.ply": outside.as_uri(),
            "FS/surf/mriscalp.stl": mri_scalp.as_uri(),
            "VEP_DS-raw.fif": raw_fif_src.as_uri(),
        },
        fs_sample_url=fs_zip.as_uri(),
    )

    assert isinstance(result, SampleDataset)
    assert result.subject_name == "FS"


def test_build_bem_writes_solution_with_inferred_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subject_dir = tmp_path / "subjects" / "FS"
    subject_dir.mkdir(parents=True)
    calls: dict[str, object] = {}

    def fake_make_bem_model(*, subject, conductivity, subjects_dir, verbose):
        calls["subject"] = subject
        calls["conductivity"] = conductivity
        calls["subjects_dir"] = subjects_dir
        return {"subject": subject}

    def fake_make_bem_solution(model, *, verbose):
        calls["model"] = model
        return {"solution": model}

    def fake_write_bem_solution(path, solution, *, overwrite, verbose):
        calls["path"] = path
        Path(path).write_text("bem", encoding="utf-8")

    monkeypatch.setattr(mne, "make_bem_model", fake_make_bem_model)
    monkeypatch.setattr(mne, "make_bem_solution", fake_make_bem_solution)
    monkeypatch.setattr(mne, "write_bem_solution", fake_write_bem_solution)

    bem_path = build_bem(subject_dir)

    assert bem_path == subject_dir / "bem" / "FS-bem-sol.fif"
    assert Path(calls["path"]).exists()
    assert calls["subject"] == "FS"
    assert calls["subjects_dir"] == str(subject_dir.parent)


def test_manifest_requires_sfreq_hz_for_matrix_inputs(tmp_path: Path) -> None:
    subject_dir = tmp_path / "R9999"
    (subject_dir / "FS").mkdir(parents=True)
    source_fif = tmp_path / "preprocessed_R9999_Run01_raw.fif"
    trans_path = _write_dummy_trans(tmp_path / "R9999_Run01_trans.fif")
    matrix_path = tmp_path / "rest_run01.csv"
    _write_raw_fif(source_fif)
    _write_sensor_matrix_csv(matrix_path)

    manifest = {
        "subjects": [
            {
                "subject": "R9999",
                "subject_dir": str(subject_dir),
                "fs_subject": "FS",
                "items": [
                    {
                        "name": "rest_run01",
                        "matrix_path": str(matrix_path),
                        "source_file": str(source_fif),
                        "trans_path": str(trans_path),
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'sfreq_hz'"):
        load_subject_specs_from_manifest(str(manifest_path))


def test_manifest_requires_time_start_s_for_matrix_inputs(tmp_path: Path) -> None:
    subject_dir = tmp_path / "R9999"
    (subject_dir / "FS").mkdir(parents=True)
    source_fif = tmp_path / "preprocessed_R9999_Run01_raw.fif"
    trans_path = _write_dummy_trans(tmp_path / "R9999_Run01_trans.fif")
    matrix_path = tmp_path / "rest_run01.csv"
    _write_raw_fif(source_fif)
    _write_sensor_matrix_csv(matrix_path)

    manifest = {
        "subjects": [
            {
                "subject": "R9999",
                "subject_dir": str(subject_dir),
                "fs_subject": "FS",
                "items": [
                    {
                        "name": "rest_run01",
                        "matrix_path": str(matrix_path),
                        "sfreq_hz": 200.0,
                        "source_file": str(source_fif),
                        "trans_path": str(trans_path),
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'time_start_s'"):
        load_subject_specs_from_manifest(str(manifest_path))


def test_build_subject_spec_from_fif_dir_uses_shared_trans_in_input_dir(
    tmp_path: Path,
) -> None:
    subject_dir = tmp_path / "R9999"
    (subject_dir / "FS").mkdir(parents=True)
    fif_dir = tmp_path / "inputs"
    fif_dir.mkdir()
    fif_a = fif_dir / "preprocessed_R9999_Run01_raw.fif"
    fif_b = fif_dir / "preprocessed_R9999_Run02_raw.fif"
    shared_trans = _write_dummy_trans(fif_dir / "R9999_trans.fif")
    _write_raw_fif(fif_a)
    _write_raw_fif(fif_b)

    spec = export_cli._build_subject_spec_from_fif_dir(
        subject_dir=str(subject_dir),
        fif_dir=str(fif_dir),
    )

    assert spec.subject == "R9999"
    assert spec.subject_dir == str(subject_dir.resolve())
    assert spec.fs_subject == "FS"
    assert [item.name for item in spec.items] == [
        "preprocessed_R9999_Run01_raw",
        "preprocessed_R9999_Run02_raw",
    ]
    assert [Path(item.fif_path).name for item in spec.items] == [
        "preprocessed_R9999_Run01_raw.fif",
        "preprocessed_R9999_Run02_raw.fif",
    ]
    assert all(item.trans_path == str(shared_trans.resolve()) for item in spec.items)


def test_export_cli_supports_convenience_fif_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject_dir = tmp_path / "R9999"
    (subject_dir / "FS").mkdir(parents=True)
    fif_dir = tmp_path / "inputs"
    fif_dir.mkdir()
    fif_path = fif_dir / "preprocessed_R9999_Run01_raw.fif"
    shared_trans = _write_dummy_trans(fif_dir / "R9999_trans.fif")
    _write_raw_fif(fif_path)

    captured: dict[str, object] = {}

    def _fake_export_subject_sensor_matrices(*, subject_spec, out_root, config, output_format):
        captured["subject_spec"] = subject_spec
        captured["out_root"] = out_root
        captured["output_format"] = output_format
        return str(Path(out_root) / subject_spec.subject)

    monkeypatch.setattr(
        export_cli,
        "export_subject_sensor_matrices",
        _fake_export_subject_sensor_matrices,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opm-source-manifest-export",
            "--subject-dir",
            str(subject_dir),
            "--fif-dir",
            str(fif_dir),
            "--out-dir",
            str(tmp_path / "exports"),
            "--output-format",
            "npz",
        ],
    )

    assert export_cli.main() == 0

    subject_spec = captured["subject_spec"]
    assert isinstance(subject_spec, export_cli.SubjectProjectionSpec)
    assert subject_spec.subject == "R9999"
    assert len(subject_spec.items) == 1
    assert subject_spec.items[0].fif_path == str(fif_path.resolve())
    assert subject_spec.items[0].trans_path == str(shared_trans.resolve())
    assert captured["output_format"] == "npz"


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

    monkeypatch.setattr(
        "opm_source_toolbox.sensor_to_roi.setup_subject_source_space",
        lambda **_: "src",
    )
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

    monkeypatch.setattr(
        "opm_source_toolbox.sensor_to_roi.project_sensor_item_to_atlas_rois",
        _fake_project,
    )

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


def test_export_manifest_to_rois_supports_npz_output(
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
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        "opm_source_toolbox.sensor_to_roi.setup_subject_source_space",
        lambda **_: "src",
    )
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
        )

    monkeypatch.setattr(
        "opm_source_toolbox.sensor_to_roi.project_sensor_item_to_atlas_rois",
        _fake_project,
    )

    out_paths = export_manifest_to_rois(
        manifest_path=str(manifest_path),
        out_root=str(tmp_path / "exports"),
        config=SourceProjectionConfig(),
        output_format="npz",
    )

    assert len(out_paths) == 1
    subject_out = Path(out_paths[0])
    out_npz = subject_out / "run01_window.npz"
    meta_path = subject_out / "metadata.json"
    assert out_npz.exists()
    assert meta_path.exists()

    with np.load(out_npz, allow_pickle=False) as payload:
        assert payload["name_col"].item() == "roi_name"
        assert payload["names"].astype(str).tolist() == ["ROI_A", "ROI_B"]
        assert payload["data"].shape == (2, 4)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["output_format"] == "npz"
    assert metadata["items"][0]["output_format"] == "npz"
    assert metadata["items"][0]["output_file"] == "run01_window.npz"
    assert metadata["items"][0]["output_csv"] is None


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
    pytest.importorskip("nilearn")
    pytest.importorskip("PIL")
    from PIL import Image

    from opm_source_toolbox.roi_surface_render import render_roi_value_map_to_surface

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
