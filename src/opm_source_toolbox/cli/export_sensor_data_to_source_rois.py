"""Project arbitrary registered sensor-space MEG data into atlas ROI time series."""

from __future__ import annotations

import argparse
import glob
import json
import os

from opm_source_toolbox.core import resolve_trans_path
from opm_source_toolbox.sensor_to_roi import (
    SensorMatrixSpec,
    SourceProjectionConfig,
    SubjectProjectionSpec,
    export_subject_sensor_matrices,
    load_subject_specs_from_manifest,
)


def _alignment_qc_subject_dir(subject_out: str, alignment_qc_out_dir: str | None, subject: str) -> str:
    if alignment_qc_out_dir:
        return os.path.join(alignment_qc_out_dir, subject)
    return os.path.join(subject_out, "alignment_qc")


def _write_alignment_qc_records(subject_spec, image_size: int, out_dir: str) -> list[dict]:
    from opm_source_toolbox.alignment_qc import render_alignment_qc_bundle

    records: list[dict] = []
    os.makedirs(out_dir, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    for item in subject_spec.items:
        geometry_info_file = item.geometry_info_file or item.source_file or item.fif_path
        if geometry_info_file is None:
            continue
        key = (str(geometry_info_file), str(item.trans_path))
        if key in seen:
            continue
        seen.add(key)

        run = item.metadata.get("run")
        if isinstance(run, int):
            label = f"run{int(run):02d}"
        else:
            label = str(item.name)

        result = render_alignment_qc_bundle(
            subject_dir=subject_spec.subject_dir,
            fs_subject=subject_spec.fs_subject,
            geometry_info_file=str(geometry_info_file),
            trans_path=str(item.trans_path),
            out_dir=out_dir,
            stem=f"{label}_sensor_head_alignment",
            title=f"{subject_spec.subject} {label} sensor/head alignment",
            image_size=int(image_size),
        )
        records.append(
            {
                "label": label,
                "geometry_info_file": os.path.abspath(str(geometry_info_file)),
                "trans_path": os.path.abspath(str(item.trans_path)),
                "montage_png": os.path.abspath(result.montage_path),
                "view_pngs": {view: os.path.abspath(path) for view, path in result.view_paths.items()},
            }
        )
    return records


def _append_alignment_qc_metadata(subject_out: str, alignment_qc_dir: str | None, alignment_qc_records: list[dict]) -> None:
    meta_path = os.path.join(subject_out, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["alignment_qc_dir"] = None if alignment_qc_dir is None else os.path.abspath(alignment_qc_dir)
    payload["alignment_qc"] = alignment_qc_records
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _is_trans_fif_path(path: str) -> bool:
    base = os.path.basename(path).lower()
    return base.endswith("_trans.fif") or base.endswith("-trans.fif")


def _collect_input_fif_paths(fif_dir: str, fif_glob: str) -> list[str]:
    pattern = os.path.join(os.path.abspath(fif_dir), fif_glob)
    paths = sorted(
        path
        for path in glob.glob(pattern, recursive=True)
        if os.path.isfile(path) and not _is_trans_fif_path(path)
    )
    if not paths:
        raise FileNotFoundError(
            f"No FIF inputs matched {pattern}; adjust --fif-dir or --fif-glob"
        )
    return [os.path.abspath(path) for path in paths]


def _default_item_name(fif_path: str) -> str:
    base = os.path.basename(fif_path)
    if base.lower().endswith(".fif"):
        base = base[:-4]
    return base or "item"


def _resolve_input_trans_path(
    *,
    fif_path: str,
    subject_dir: str,
    trans_path: str | None,
) -> str:
    if trans_path is not None:
        if not os.path.exists(trans_path):
            raise FileNotFoundError(f"Missing trans file: {trans_path}")
        return os.path.abspath(trans_path)

    search_dirs = [os.path.abspath(os.path.dirname(fif_path))]
    subject_dir_abs = os.path.abspath(subject_dir)
    if subject_dir_abs not in search_dirs:
        search_dirs.append(subject_dir_abs)

    last_missing: FileNotFoundError | None = None
    for search_dir in search_dirs:
        try:
            return os.path.abspath(resolve_trans_path(search_dir, fif_path))
        except FileNotFoundError as exc:
            last_missing = exc
            continue

    if last_missing is not None:
        raise last_missing
    raise FileNotFoundError(f"Could not resolve trans file for FIF input {fif_path}")


def _build_subject_spec_from_fif_dir(
    *,
    subject_dir: str,
    fif_dir: str,
    subject: str | None = None,
    fs_subject: str = "FS",
    fif_glob: str = "*.fif",
    trans_path: str | None = None,
) -> SubjectProjectionSpec:
    subject_dir_abs = os.path.abspath(subject_dir)
    fif_dir_abs = os.path.abspath(fif_dir)
    if not os.path.isdir(subject_dir_abs):
        raise FileNotFoundError(f"Missing subject_dir: {subject_dir_abs}")
    if not os.path.isdir(os.path.join(subject_dir_abs, fs_subject)):
        raise FileNotFoundError(
            f"Missing FreeSurfer anatomy under {subject_dir_abs}: {fs_subject}"
        )
    if not os.path.isdir(fif_dir_abs):
        raise FileNotFoundError(f"Missing fif_dir: {fif_dir_abs}")

    subject_name = subject or os.path.basename(subject_dir_abs.rstrip(os.sep)) or "subject"
    fif_paths = _collect_input_fif_paths(fif_dir_abs, fif_glob)
    items = [
        SensorMatrixSpec(
            name=_default_item_name(fif_path),
            fif_path=fif_path,
            trans_path=_resolve_input_trans_path(
                fif_path=fif_path,
                subject_dir=subject_dir_abs,
                trans_path=trans_path,
            ),
        )
        for fif_path in fif_paths
    ]
    return SubjectProjectionSpec(
        subject=subject_name,
        subject_dir=subject_dir_abs,
        fs_subject=fs_subject,
        items=items,
    )


def _load_subject_specs_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[SubjectProjectionSpec]:
    if args.manifest:
        if args.subject_dir or args.fif_dir or args.trans_path or args.subject:
            parser.error(
                "--manifest cannot be combined with --subject, --subject-dir, "
                "--fif-dir, or --trans-path"
            )
        return load_subject_specs_from_manifest(args.manifest)

    if not args.subject_dir or not args.fif_dir:
        parser.error(
            "pass --manifest, or use convenience FIF mode with both "
            "--subject-dir and --fif-dir"
        )
    return [
        _build_subject_spec_from_fif_dir(
            subject_dir=args.subject_dir,
            fif_dir=args.fif_dir,
            subject=args.subject,
            fs_subject=args.fs_subject,
            fif_glob=args.fif_glob,
            trans_path=args.trans_path,
        )
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=None,
        help="JSON manifest describing subject directories and CSV/FIF sensor items",
    )
    ap.add_argument(
        "--subject",
        default=None,
        help="Subject identifier for convenience FIF mode; defaults to basename(subject_dir)",
    )
    ap.add_argument(
        "--subject-dir",
        default=None,
        help="Subject directory containing the FreeSurfer folder for convenience FIF mode",
    )
    ap.add_argument(
        "--fs-subject",
        default="FS",
        help="FreeSurfer folder name under subject_dir for convenience FIF mode; defaults to FS",
    )
    ap.add_argument(
        "--fif-dir",
        default=None,
        help="Directory containing FIF inputs for convenience FIF mode",
    )
    ap.add_argument(
        "--fif-glob",
        default="*.fif",
        help="Glob for FIF inputs under --fif-dir in convenience FIF mode; defaults to *.fif",
    )
    ap.add_argument(
        "--trans-path",
        default=None,
        help="Optional explicit trans file to use for all FIF inputs in convenience FIF mode",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.getcwd(), "source_roi_exports"),
        help="Output root for exported ROI time-series files; defaults to ./source_roi_exports",
    )
    ap.add_argument(
        "--output-format",
        choices=("csv", "npz"),
        default="csv",
        help="Format for exported ROI time-series files; defaults to csv",
    )
    ap.add_argument("--write-alignment-qc", action="store_true", help="Write optional multi-view sensor/head alignment QC PNGs during export")
    ap.add_argument("--alignment-qc-out-dir", default=None, help="Optional output root for alignment QC PNGs; defaults to <subject-export>/alignment_qc")
    ap.add_argument("--alignment-qc-image-size", type=int, default=320, help="Per-view image size in pixels for alignment QC montages")
    ap.add_argument("--atlas-parc", default="Schaefer2018_200Parcels_7Networks_order")
    ap.add_argument("--atlas-subject", default=None)
    ap.add_argument("--atlas-subjects-dir", default=None)
    ap.add_argument("--source-spacing", default="ico3")
    ap.add_argument("--inverse-kind", choices=("mne", "lcmv"), default="mne")
    ap.add_argument("--mne-method", choices=("MNE", "dSPM", "sLORETA", "eLORETA"), default="MNE")
    ap.add_argument("--mne-pick-ori", choices=("normal", "none"), default="normal")
    ap.add_argument("--label-mode", default="pca_flip")
    ap.add_argument("--estimate-covariance", action="store_true", help="Estimate empirical covariance from the provided sensor data items")
    ap.add_argument("--covariance-scope", choices=("per_item", "per_subject"), default="per_item", help="When estimating covariance, use each item separately or all subject items together")
    ap.add_argument("--covariance-fallback", choices=("identity", "adhoc"), default="identity", help="Fallback covariance used when --estimate-covariance is not enabled")
    ap.add_argument("--snr", type=float, default=3.0)
    ap.add_argument("--loose", type=float, default=0.2)
    ap.add_argument("--depth", type=float, default=0.8)
    ap.add_argument("--beamformer-reg", type=float, default=0.05)
    ap.add_argument("--beamformer-pick-ori", choices=("normal", "max-power", "none"), default="normal")
    ap.add_argument("--beamformer-weight-norm", choices=("unit-noise-gain-invariant", "unit-noise-gain", "nai", "none"), default="unit-noise-gain-invariant")
    ap.add_argument("--beamformer-depth", type=float, default=None)
    ap.add_argument("--conductor-kind", choices=("auto", "bem", "sphere"), default="auto", help="Use BEM if present, otherwise fall back to a spherical conductor")
    ap.add_argument("--sphere-origin-x", type=float, default=0.0)
    ap.add_argument("--sphere-origin-y", type=float, default=0.0)
    ap.add_argument("--sphere-origin-z", type=float, default=0.0)
    ap.add_argument("--sphere-head-radius", type=float, default=0.07)

    args = ap.parse_args()

    config = SourceProjectionConfig(
        atlas_parc=args.atlas_parc,
        atlas_subject=args.atlas_subject,
        atlas_subjects_dir=args.atlas_subjects_dir,
        source_spacing=args.source_spacing,
        inverse_kind=args.inverse_kind,
        mne_method=args.mne_method,
        mne_pick_ori=args.mne_pick_ori,
        label_mode=args.label_mode,
        estimate_covariance=bool(args.estimate_covariance),
        covariance_scope=args.covariance_scope,
        covariance_fallback=args.covariance_fallback,
        snr=float(args.snr),
        loose=float(args.loose),
        depth=float(args.depth),
        beamformer_reg=float(args.beamformer_reg),
        beamformer_pick_ori=args.beamformer_pick_ori,
        beamformer_weight_norm=args.beamformer_weight_norm,
        beamformer_depth=args.beamformer_depth,
        conductor_kind=args.conductor_kind,
        sphere_origin=(float(args.sphere_origin_x), float(args.sphere_origin_y), float(args.sphere_origin_z)),
        sphere_head_radius=float(args.sphere_head_radius),
    )

    subject_specs = _load_subject_specs_from_args(args, ap)
    os.makedirs(args.out_dir, exist_ok=True)
    for subject_spec in subject_specs:
        subject_out = export_subject_sensor_matrices(
            subject_spec=subject_spec,
            out_root=args.out_dir,
            config=config,
            output_format=args.output_format,
        )
        if not args.write_alignment_qc:
            continue
        alignment_qc_dir = _alignment_qc_subject_dir(subject_out=subject_out, alignment_qc_out_dir=args.alignment_qc_out_dir, subject=subject_spec.subject)
        try:
            records = _write_alignment_qc_records(subject_spec=subject_spec, image_size=int(args.alignment_qc_image_size), out_dir=alignment_qc_dir)
            _append_alignment_qc_metadata(subject_out=subject_out, alignment_qc_dir=alignment_qc_dir, alignment_qc_records=records)
            print(f"Alignment QC plots: {alignment_qc_dir}")
        except Exception as exc:
            print(f"Alignment QC failed for {subject_spec.subject}: {exc}")

    print("Generic sensor-to-ROI export finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
