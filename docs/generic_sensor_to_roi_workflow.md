# Generic Sensor-To-ROI Workflow

This workflow implements the reusable stage:

`registered sensor-space MEG matrix + trans + anatomy + atlas -> ROI time series`

It is intentionally independent of triggers, epochs, and experiment-specific condition
logic. You can feed it either an already-prepared sensor matrix or a direct FIF input,
for example:

- a resting-state run
- a continuous data segment
- an epoched average
- a task-condition average produced by some other pipeline
- a `Raw` / `Epochs` / `Evoked` FIF that should be converted on the fly

## Quickstart: One Subject, One FIF

For the happiest-path local case, assume:

- one subject directory such as `/data/co-reg/R9999/` containing `/data/co-reg/R9999/FS/`
- one input FIF such as `/data/opm/R9999/run01_raw.fif`
- one matching transform in that same directory such as `/data/opm/R9999/run01_trans.fif`

First, make sure the default Schaefer atlas is available:

```bash
uv run opm-source-fetch-atlas --atlas-name schaefer
```

Then run the export directly from the FIF directory:

```bash
uv run opm-source-manifest-export \
  --subject-dir /data/co-reg/R9999 \
  --fif-dir /data/opm/R9999 \
  --out-dir /data/source_roi_exports
```

This convenience mode skips the manifest. The exporter scans `--fif-dir`, ignores
`*_trans.fif` files, and because there is exactly one transform file in that directory
it reuses that transform automatically for the input FIF.

Outputs are written under `/data/source_roi_exports/R9999/`. By default the ROI matrix
is written as `.csv`; add `--output-format npz` for compressed binary output.

## CLI

Use:

`uv run opm-source-manifest-export --manifest path/to/manifest.json`

Add `--output-format npz` to write compressed binary ROI matrices instead of CSV.

or:

`./.venv/bin/python -m opm_source_toolbox.cli.export_sensor_data_to_source_rois --manifest path/to/manifest.json`

Convenience FIF mode for one subject:

`uv run opm-source-manifest-export --subject-dir /path/to/co-reg/R9999 --fif-dir /path/to/fifs`

or:

`./.venv/bin/python -m opm_source_toolbox.cli.export_sensor_data_to_source_rois --subject-dir /path/to/co-reg/R9999 --fif-dir /path/to/fifs`

In convenience mode the CLI scans `--fif-dir` for FIF inputs, ignores `*_trans.fif`
files, and auto-resolves one `trans` per FIF. If a FIF directory contains exactly one
`*_trans.fif`, that file is reused for all FIF inputs in that directory. If not, the
CLI falls back to searching `--subject-dir` with the same naming rules.

## Input Contract

The workflow reads a JSON manifest with one or more subject blocks. Each subject block
declares:

- `subject`
- `subject_dir`
  Directory containing the subject FreeSurfer folder, for example `<subject_dir>/FS/`
- `fs_subject`
  Usually `FS` in the current project
- `items`
  A list of sensor matrices to project into source space

Each item declares:

- `name`
- `trans_path`
  Head-to-MRI transform for that sensor matrix
- exactly one of:
  - `matrix_path`
    Path to a CSV sensor matrix with columns `ch_name, t000, t001, ...`
  - `fif_path`
    Path to a `Raw`, `Epochs`, or `Evoked` FIF file
- `sfreq_hz` (required with `matrix_path`)
  Sampling rate for that matrix input. FIF-backed inputs carry this in the file header.
- `time_start_s` (required with `matrix_path`)
  Time in seconds for the first sample in that matrix. FIF-backed inputs carry this in
  the object time axis.
- `source_file` (optional for direct FIF input)
  FIF used to recover channel geometry / header info
- `geometry_info_file` (optional)
  Override file used for channel geometry if different from `source_file`
- `metadata` (optional)
  Arbitrary JSON copied into the output metadata

## Minimal Manifest Example

```json
{
  "subjects": [
    {
      "subject": "R9999",
      "subject_dir": "/path/to/co-reg/R9999",
      "fs_subject": "FS",
      "items": [
        {
          "name": "rest_run01",
          "matrix_path": "/path/to/rest_run01.csv",
          "sfreq_hz": 200.0,
          "time_start_s": 0.0,
          "source_file": "/path/to/preprocessed_R9999_Run01.fif",
          "trans_path": "/path/to/R9999_Run01_trans.fif"
        },
        {
          "name": "task_average_left",
          "matrix_path": "/path/to/task_average_left.csv",
          "sfreq_hz": 200.0,
          "time_start_s": -0.1,
          "source_file": "/path/to/preprocessed_R9999_Run02.fif",
          "trans_path": "/path/to/R9999_Run02_trans.fif",
          "metadata": {
            "condition": "left_hand_average"
          }
        }
      ]
    }
  ]
}
```

Direct FIF example:

```json
{
  "subjects": [
    {
      "subject": "R9999",
      "subject_dir": "/path/to/co-reg/R9999",
      "fs_subject": "FS",
      "items": [
        {
          "name": "rest_raw_window",
          "fif_path": "/path/to/preprocessed_R9999_Run01_raw.fif",
          "fif_kind": "raw",
          "tmin_s": 10.0,
          "tmax_s": 20.0,
          "trans_path": "/path/to/R9999_Run01_trans.fif"
        },
        {
          "name": "averaged_epochs",
          "fif_path": "/path/to/R9999-epo.fif",
          "fif_kind": "epochs",
          "epochs_average": true,
          "trans_path": "/path/to/R9999_Run02_trans.fif"
        }
      ]
    }
  ]
}
```

There is no checked-in manifest example in this repo beyond the snippets above. For the
simplest local workflow, prefer convenience FIF mode and skip the manifest entirely.

## Covariance Flag

Covariance estimation is controlled by:

`--estimate-covariance`

Behavior:

- without the flag:
  - minimum norm uses the fallback covariance from `--covariance-fallback`
  - default fallback is `identity`
- with the flag:
  - an empirical covariance is estimated from the provided sensor data item
  - `--covariance-scope per_item` estimates it separately for each item
  - `--covariance-scope per_subject` estimates one covariance across all items for that subject
  - `per_subject` currently requires identical channel ordering across those items

Current constraint:

- `lcmv` requires `--estimate-covariance`

This keeps the generic source workflow free of hard-coded baseline or trigger logic.
If you later want baseline-specific covariance, that should be added as a higher-level
option on top of this generic stage.

## Outputs

Outputs are written to:

`<out-dir>/<SUBJECT>/`

Files:

- one ROI matrix file per input item (`.csv` by default, or `.npz` with `--output-format npz`)
- `metadata.json` describing atlas, inverse settings, covariance choices, and provenance

CSV outputs use:

- `roi_name`
- `t000, t001, ...`

NPZ outputs store:

- `name_col`
- `names`
- `data`
