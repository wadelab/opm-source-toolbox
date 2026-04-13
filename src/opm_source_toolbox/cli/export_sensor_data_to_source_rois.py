"""Project arbitrary registered sensor-space MEG data into atlas ROI time series."""

from __future__ import annotations

import argparse
import json
import os

from opm_source_toolbox.alignment_qc import render_alignment_qc_bundle
from opm_source_toolbox.sensor_to_roi import (
    SourceProjectionConfig,
    export_subject_sensor_matrices,
    load_subject_specs_from_manifest,
)


def _alignment_qc_subject_dir(subject_out: str, alignment_qc_out_dir: str | None, subject: str) -> str:
    if alignment_qc_out_dir:
        return os.path.join(alignment_qc_out_dir, subject)
    return os.path.join(subject_out, "alignment_qc")


def _write_alignment_qc_records(subject_spec, image_size: int, out_dir: str) -> list[dict]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON manifest describing subject directories and CSV/FIF sensor items")
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.getcwd(), "source_roi_exports"),
        help="Output root for exported ROI time-series CSVs; defaults to ./source_roi_exports",
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

    subject_specs = load_subject_specs_from_manifest(args.manifest)
    os.makedirs(args.out_dir, exist_ok=True)
    for subject_spec in subject_specs:
        subject_out = export_subject_sensor_matrices(subject_spec=subject_spec, out_root=args.out_dir, config=config)
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