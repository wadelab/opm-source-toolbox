# `render_alignment_qc.py`

This is the generic alignment-QC helper for the source toolbox.

It converts:

`geometry/header FIF + trans + subject anatomy -> sensor/head alignment montage`

Unlike the vibro-specific montage builder, this script does not depend on:

- exported run/condition CSVs
- subject lists inferred from repo outputs
- vibrotactile task structure

It renders one subject and one geometry/trans pairing at a time.

## CLI

Run with:

```bash
uv run opm-source-alignment-qc \
  --subject-dir /path/to/co-reg/R9999 \
  --geometry-info-file /path/to/preprocessed_R9999_Run01_raw.fif \
  --trans-path /path/to/R9999_Run01_trans.fif
```

or:

```bash
./.venv/bin/python render_alignment_qc.py \
  --subject-dir /path/to/co-reg/R9999 \
  --geometry-info-file /path/to/preprocessed_R9999_Run01_raw.fif \
  --trans-path /path/to/R9999_Run01_trans.fif
```

If you do not pass `--out-dir`, renders default to `./alignment_qc`
relative to the current working directory.

## Required Inputs

- `--subject-dir`
  Subject co-registration directory containing the FreeSurfer subject folder
- `--geometry-info-file`
  FIF file used to recover MEG channel geometry/header information
- `--trans-path`
  Head-to-MRI transform for that geometry file

## Useful Flags

- `--fs-subject`
- `--out-dir`
- `--stem`
- `--title`
- `--views`
- `--image-size`

Default views are:

- `oblique`
- `side`
- `top`

## Outputs

The script writes:

- one PNG per requested view
- one combined montage PNG

With the default CLI settings, those files are written under:

```text
./alignment_qc/
```

## Python API

The generic renderer is also available from the toolbox package:

- [alignment_qc.py](/raid/toolbox/git/vibroMEG/opm_source_toolbox/alignment_qc.py)

Main entrypoints:

- `render_alignment_screenshot(...)`
- `render_alignment_qc_bundle(...)`