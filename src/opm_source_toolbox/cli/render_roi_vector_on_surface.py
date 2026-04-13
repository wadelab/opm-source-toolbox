"""Render a simple ROI-name/value table onto the atlas surface."""

from __future__ import annotations

import argparse
import os

from opm_source_toolbox.roi_surface_render import (
    SurfaceRenderConfig,
    load_roi_value_map_csv,
    render_roi_value_map_to_surface,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, help="CSV containing ROI names and one value column")
    ap.add_argument("--roi-name-col", default="roi_name")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--out-dir", default=os.path.join(os.getcwd(), "roi_surface_renders"), help="Output directory for rendered surface PNGs; defaults to ./roi_surface_renders")
    ap.add_argument("--stem", default=None, help="Filename stem for the written PNGs; defaults to <csv-basename>_<value-col>")
    ap.add_argument("--title", default=None)
    ap.add_argument("--atlas-parc", default="Schaefer2018_200Parcels_7Networks_order")
    ap.add_argument("--atlas-subject", default=None)
    ap.add_argument("--atlas-subjects-dir", default=None)
    ap.add_argument("--surface", default="inflated")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--color-mode", choices=("auto", "positive", "symmetric"), default="auto", help="Auto-detect signed versus positive-only scaling, or force one mode")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=1e-12)
    ap.add_argument("--bg-binary-low", type=float, default=0.44)
    ap.add_argument("--bg-binary-high", type=float, default=0.56)

    args = ap.parse_args()
    stem = args.stem
    if stem is None:
        base = os.path.splitext(os.path.basename(args.in_csv))[0]
        stem = f"{base}_{args.value_col}"
    title = args.title or f"{stem} surface map"
    config = SurfaceRenderConfig(
        atlas_parc=args.atlas_parc,
        atlas_subject=args.atlas_subject,
        atlas_subjects_dir=args.atlas_subjects_dir,
        surface=args.surface,
        cmap=args.cmap,
        color_mode=args.color_mode,
        vmin=args.vmin,
        vmax=args.vmax,
        threshold=float(args.threshold),
        bg_binary_low=float(args.bg_binary_low),
        bg_binary_high=float(args.bg_binary_high),
    )
    value_by_name = load_roi_value_map_csv(csv_path=args.in_csv, roi_name_col=args.roi_name_col, value_col=args.value_col)
    result = render_roi_value_map_to_surface(value_by_name=value_by_name, out_dir=args.out_dir, stem=stem, title=title, config=config)
    print("ROI vector surface render finished.")
    print(f"  Montage: {result['montage_path']}")
    print(f"  vmin={result['vmin']:.3g} vmax={result['vmax']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())