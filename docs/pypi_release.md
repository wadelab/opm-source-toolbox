# PyPI Release

This repository is prepared for PyPI trusted publishing.

## Package Identity

- distribution name: `opm-source-roi`
- import package: `opm_source_toolbox`
- current public repository: `https://github.com/wadelab/opm-source-toolbox`

## One-Time PyPI Setup

Before the GitHub workflow can publish, create the project on PyPI and configure a
trusted publisher.

Recommended trusted publisher settings:

- owner: `wadelab`
- repository: `opm-source-toolbox`
- workflow: `publish-pypi.yml`
- environment: `pypi`

The publish workflow in this repository expects that trusted publisher setup to exist.

## Local Preflight

Run before creating a release:

```bash
uv run --extra dev python -m pytest -q tests/test_opm_source_toolbox.py
uv run --with build python -m build
uv run --with twine python -m twine check dist/*
```

## Release Flow

1. Update `version` in `pyproject.toml`.
2. Commit the version bump.
3. Create and push a matching git tag.
4. Create a GitHub release from that tag, or run the publish workflow manually.
5. GitHub Actions will build the distributions and publish them to PyPI.

## After Publish

Downstream projects can switch from a git pin to a normal dependency such as:

```bash
uv add opm-source-roi==0.1.0
```