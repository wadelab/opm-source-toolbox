"""Render a generic multi-view sensor/head alignment QC bundle."""

from __future__ import annotations

import argparse
import os

from opm_source_toolbox.alignment_qc import render_alignment_qc_bundle


def _default_stem(path: str) -> str:
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    if stem.endswith("_raw"):
        stem = stem[:-4]
    return stem or "alignment_qc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-dir", required=True, help="Subject co-registration directory containing the FreeSurfer subject folder")
    ap.add_argument("--fs-subject", default="FS")
    ap.add_argument("--geometry-info-file", required=True, help="FIF file used to recover MEG sensor geometry/header information")
    ap.add_argument("--trans-path", required=True, help="Head-to-MRI transform for this data item")
    ap.add_argument("--out-dir", default=os.path.join(os.getcwd(), "alignment_qc"), help="Output directory for alignment QC PNGs; defaults to ./alignment_qc")
    ap.add_argument("--stem", default=None, help="Filename stem for written PNGs; defaults to the geometry-info basename")
    ap.add_argument("--title", default=None)
    ap.add_argument("--views", nargs="+", default=["oblique", "side", "top"], help="Views to render; defaults to oblique side top")
    ap.add_argument("--image-size", type=int, default=320)

    args = ap.parse_args()
    stem = args.stem or _default_stem(args.geometry_info_file)
    title = args.title or f"{stem} sensor/head alignment"
    result = render_alignment_qc_bundle(
        subject_dir=args.subject_dir,
        fs_subject=args.fs_subject,
        geometry_info_file=args.geometry_info_file,
        trans_path=args.trans_path,
        out_dir=args.out_dir,
        stem=stem,
        title=title,
        views=args.views,
        image_size=int(args.image_size),
    )
    print("Alignment QC render finished.")
    print(f"  Montage: {result.montage_path}")
    for view, path in result.view_paths.items():
        print(f"  {view}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())