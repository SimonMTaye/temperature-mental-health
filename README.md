

# Setup

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Run `uv sync` to fetch all dependencies

## Cleaning

To run the cleaning pipeline:

`uv run clean`

Note: this requires access to the raw data


## Deploying the Website

The repository uses "hooks" or simple scripts, that run every time you commit to
render the preview of results and then share it online. Below are instructions 
on setting it up


Enable the repo-managed pre-commit hook once per clone:

```sh
git config core.hooksPath .githooks
```

The hook renders `code/preview/preview.qmd` to `docs/index.html` locally and
stages the rendered page with `git add docs/index.html`. GitHub Pages serves this
as the root preview page for the site and PR previews.


# Sources

- GADM file (geographic boundaries) download from this link:
  - https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_IDN.gpkg
