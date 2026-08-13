# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[web,dev]"
pytest
```

## Before opening a pull request

```bash
ruff check .        # lint
pytest              # 54 tests
```

CI runs both on Python 3.11 and 3.12, and builds and smoke-tests the Docker
image.

## Working on the model

The feature order in `FEATURE_NAMES` is part of the saved artifact's contract.
**Appending** a feature is fine; reordering or removing one invalidates every
existing model, so bump `ARTIFACT_VERSION` in `src/cosmoquake/model.py` if you
do.

Retrain and inspect with:

```bash
cosmoquake train
cosmoquake info
```

If you change the model, update `docs/MODEL_CARD.md` with the new numbers from
`cosmoquake info`. Report accuracy alongside the majority-class baseline — on
this catalog a constant predictor scores 84.2%, so a bare accuracy figure is
misleading. Include per-class recall and the rare-class AUCs.

Claims of improved rare-class performance need cross-validated evidence. With
three shallow moonquakes in the catalog, a single lucky fold is not a result.

## Adding data

The biggest available improvement to this project is more labelled events.
Apollo 14, 15 and 16 catalogs would roughly quadruple the training set and are
the most direct path to a classifier that works on the rare classes.

New rows go in `data/moonquakes.csv` with the columns in `FEATURE_NAMES` plus
`label`. Generate them by running the feature extractor over catalogued
waveforms rather than by hand.

## Style

- Follow the surrounding code: type hints, Google-style docstrings, `ruff`
  defaults at 100 columns.
- Comments should explain *why*, not restate the code. The existing comments
  about sample-index conversion and the travelling scaler are the model here —
  each one marks a bug that was actually hit.
- Tests that encode a fixed bug should say so in the docstring.

## Notebooks

`notebooks/` holds the original hackathon submission as a historical record.
They are excluded from lint and not maintained — port changes into
`src/cosmoquake/` instead.
