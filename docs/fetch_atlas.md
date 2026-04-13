# `fetch_atlas.py`

This helper imports atlas annotation files into a reusable subjects directory so the
toolbox can resolve them later without bundling every atlas inside the package.

It supports two modes:

- `schaefer`
  Import packaged or locally available Schaefer `.annot` files into a subjects dir
- `annotation-pair`
  Import any explicit `lh.*.annot` / `rh.*.annot` pair as a named atlas entry

## CLI

Use the packaged Schaefer annotations and copy them into the default toolbox cache:

```bash
uv run opm-source-fetch-atlas --atlas-name schaefer
```

Import Schaefer annotations from a local directory into a specific subjects dir:

```bash
uv run opm-source-fetch-atlas \
  --atlas-name schaefer \
  --atlas-parc Schaefer2018_200Parcels_7Networks_order \
  --source-dir /path/to/atlas-files \
  --atlas-subjects-dir /path/to/subjects
```

Import an arbitrary annotation pair:

```bash
uv run opm-source-fetch-atlas \
  --atlas-name annotation-pair \
  --atlas-parc CustomAtlas \
  --atlas-subject fsaverage \
  --lh-annot-path /path/to/lh.CustomAtlas.annot \
  --rh-annot-path /path/to/rh.CustomAtlas.annot
```

## Default Cache Location

If you do not pass `--atlas-subjects-dir`, the helper installs atlas files into the
toolbox cache under:

```text
$XDG_CACHE_HOME/opm_source_toolbox/subjects
```

or, if `XDG_CACHE_HOME` is unset:

```text
~/.cache/opm_source_toolbox/subjects
```

Atlas resolution now searches that cache automatically, as well as:

- `--atlas-subjects-dir`
- `OPM_SOURCE_ATLAS_SUBJECTS_DIR`
- `SUBJECTS_DIR`
- `$FREESURFER_HOME/subjects`

## Python API

The helper is also available from the package:

- [atlas_fetch.py](/raid/toolbox/git/vibroMEG/opm_source_toolbox/atlas_fetch.py)

Main entrypoints:

- `fetch_atlas(...)`
- `fetch_schaefer_annotations(...)`
- `import_annotation_pair(...)`