# data/

Video and dataset assets for Kanshigan, versioned via [DVC](https://dvc.org).

## Layout

- `raw/` — original match recordings (immutable inputs). One source of truth.
- `processed/` — derived artifacts (cropped, resampled, annotated). Regeneratable from `raw/` + code.

## How it works

Files in this tree are **not stored in git**. Their content lives in a Google Drive remote and is referenced from git via `.dvc` pointer files.

See [`diario-de-bordo/09-sincronizacao-de-dados-dvc.md`](../diario-de-bordo/09-sincronizacao-de-dados-dvc.md) for setup, auth and the day-to-day workflow.

## Quick reference

```bash
dvc pull                          # download all tracked data
dvc add data/raw/match_001.mp4    # start tracking a new file
dvc push                          # upload to gdrive remote
```
