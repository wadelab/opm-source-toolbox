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

## CLI

Use:

`uv run opm-source-manifest-export --manifest path/to/manifest.json`

or:

`./.venv/bin/python export_sensor_data_to_source_rois.py --manifest path/to/manifest.json`

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
          "source_file": "/path/to/preprocessed_R9999_Run01.fif",
          "trans_path": "/path/to/R9999_Run01_trans.fif"
        },
        {
          "name": "task_average_left",
          "matrix_path": "/path/to/task_average_left.csv",
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

Checked-in example manifest:

- [generic_source_from_fif_manifest.json](/raid/toolbox/git/vibroMEG/docs/examples/generic_source_from_fif_manifest.json)

- [generic_source_from_fif_manifest.json](/raid/toolbox/git/vibroMEG/docs/examples/generic_source_from_fif_manifest.json)

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

- one ROI CSV per input item
- `metadata.json` describing atlas, inverse settings, covariance choices, and provenance

Each ROI CSV uses:

- `roi_name`
- `t000, t001, ...`
