"""Fetch or import atlas annotation files into a reusable subjects directory."""

from __future__ import annotations

import argparse

from opm_source_toolbox.atlas_fetch import fetch_atlas
from opm_source_toolbox.core import DEFAULT_ATLAS_PARC, default_atlas_subjects_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--atlas-name",
        choices=("schaefer", "annotation-pair", "annot-pair"),
        default="schaefer",
    )
    ap.add_argument("--atlas-parc", default=DEFAULT_ATLAS_PARC)
    ap.add_argument("--atlas-subject", default=None)
    ap.add_argument(
        "--atlas-subjects-dir",
        default=None,
        help="Target subjects directory for imported atlas annotations; defaults to the toolbox cache",
    )
    ap.add_argument(
        "--source-dir",
        default=None,
        help="Optional source directory containing lh.<atlas_parc>.annot and rh.<atlas_parc>.annot",
    )
    ap.add_argument("--lh-annot-path", default=None)
    ap.add_argument("--rh-annot-path", default=None)
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()
    result = fetch_atlas(
        atlas_name=args.atlas_name,
        atlas_parc=args.atlas_parc,
        atlas_subject=args.atlas_subject,
        atlas_subjects_dir=args.atlas_subjects_dir,
        source_dir=args.source_dir,
        lh_annot_path=args.lh_annot_path,
        rh_annot_path=args.rh_annot_path,
        force=bool(args.force),
    )
    print("Atlas import finished.")
    print(f"  name: {result.atlas_name}")
    print(f"  parc: {result.atlas_parc}")
    print(f"  subject: {result.atlas_subject}")
    print(f"  subjects_dir: {result.atlas_subjects_dir}")
    print(f"  label_dir: {result.label_dir}")
    print(f"  source: {result.source}")
    print("  annotation_paths:")
    for path in result.atlas_paths:
        print(f"    {path}")
    if args.atlas_subjects_dir is None:
        print(
            "  note: atlas resolution will also search this cache automatically: "
            f"{default_atlas_subjects_dir(create=True)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())