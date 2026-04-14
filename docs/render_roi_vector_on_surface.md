# `opm-source-surface-vector`

This is the generic surface-render helper for the source toolbox.

It converts:

`ROI name/value table -> fsaverage-style cortical surface montage`

Unlike the vibro-specific surface renderer, this script does not know anything about:

- runs
- conditions
- frequencies
- summary-table conventions

It just needs a table with ROI names and one numeric value column.

## Input CSV

At minimum the CSV needs:

- `roi_name`
- one numeric value column, for example `value`

Example:

```csv
roi_name,value
7Networks_LH_SomMot_1,2.8
7Networks_LH_SomMot_2,3.1
7Networks_RH_SomMot_1,1.4
7Networks_RH_DefaultPar_3,0.2
```

## CLI

Run with:

```bash
uv run opm-source-surface-vector --in-csv path/to/roi_values.csv --value-col value
```

or:

```bash
./.venv/bin/python -m opm_source_toolbox.cli.render_roi_vector_on_surface --in-csv path/to/roi_values.csv --value-col value
```

If you do not pass `--out-dir`, renders now default to `./roi_surface_renders`
relative to the current working directory instead of writing back into this repo.

## Useful Flags

- `--roi-name-col`
- `--value-col`
- `--out-dir`
- `--stem`
- `--title`
- `--atlas-parc`
- `--atlas-subject`
- `--atlas-subjects-dir`
- `--surface`
- `--cmap`
- `--color-mode {auto,positive,symmetric}`
- `--vmin`
- `--vmax`

`--color-mode auto` is usually the right default:

- positive-only vectors get a one-sided colormap
- signed vectors get a symmetric colorbar around zero

## Outputs

The script writes:

- four view PNGs
  - left lateral
  - right lateral
  - left dorsal
  - right dorsal
- one combined montage PNG

With the default CLI settings, those files are written under:

```text
./roi_surface_renders/
```

## Python API

The generic renderer is also available from the toolbox package:

- [roi_surface_render.py](../src/opm_source_toolbox/roi_surface_render.py)

Main entrypoints:

- `render_roi_vector_to_surface(roi_names, values, ...)`
- `render_roi_value_map_to_surface(value_by_name, ...)`
