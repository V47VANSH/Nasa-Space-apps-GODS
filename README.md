<div align="center">

# Cosmoquake Analyzer

**Detect, compress and classify seismic events in Apollo and InSight records.**

[![CI](https://github.com/V47VANSH/Richter_Victors-Cosmoquake_Analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/V47VANSH/Richter_Victors-Cosmoquake_Analyzer/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker ready](https://img.shields.io/badge/docker-ready-2496ed.svg)](Dockerfile)

*NASA Space Apps Challenge 2024 — Seismic Detection Across the Solar System*

</div>

---

## The problem

A planetary seismometer generates far more data than it can send home. Apollo's
network returned a continuous trickle for years; InSight's bandwidth on Mars was
tighter still. Sending raw waveforms is impossible, but the interesting part of
a record — a quake — occupies a few minutes out of hours of silence.

Cosmoquake Analyzer finds that part on-station and downlinks a summary instead
of the signal.

```
  ┌────────────┐   ┌───────────────┐   ┌──────────────┐   ┌───────────────┐
  │   read     │──▶│  STA/LTA      │──▶│  FFT feature │──▶│ random forest │
  │ csv/mseed  │   │  arrival time │   │  extraction  │   │  classifier   │
  └────────────┘   └───────────────┘   └──────────────┘   └───────────────┘
     576 KB            arrival: t=1670 s       48 bytes        impact_mq
                                                              ~12,000× smaller
```

One hour of 20 Hz InSight data is 576 KB of samples. What leaves the station is
five floating-point features plus an arrival time — **48 bytes**.

## Quickstart

```bash
git clone https://github.com/V47VANSH/Richter_Victors-Cosmoquake_Analyzer.git
cd Richter_Victors-Cosmoquake_Analyzer

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[web,dev]"

cosmoquake predict Resources/Data/mars/test/data/XB.ELYSE.02.BHV.2022-05-04HR23_evid0001.csv
```

```
XB.ELYSE.02.BHV.2022-05-04HR23_evid0001.csv
  event         impact_mq  (Meteoroid impact)
  confidence    98.3%
  arrival       1,670.1 s
  record        71,999 samples, 3,600 s @ 20.00 Hz
  downlink      575,992 B -> 12,000x smaller
  features
    max_velocity      9229.01
    weighted_velocity 1206.71
    mean_velocity     1279.17
    freq_of_max       2.44527
    area              1.16426e+08
  probabilities
    impact_mq         98.3%
    shallow_mq        1.7%
    deep_mq           0.0%
  WARNING       5 of 5 features fall outside the training range ...
```

That warning is the tool working correctly — see [Honest results](#honest-results).

## Web service

```bash
cosmoquake serve            # http://localhost:8000
```

Or with Docker:

```bash
docker compose up --build   # http://localhost:8000
```

Drag a `.csv` or `.mseed` file onto the page to get an arrival time, the
extracted features and a classification.

| Endpoint | Purpose |
|---|---|
| `GET /` | Upload interface |
| `GET /health` | Liveness probe; reports whether the model loaded |
| `GET /api/model` | Classes, feature importances, training range, CV scores |
| `POST /api/predict` | Classify an uploaded record |
| `GET /docs` | Interactive OpenAPI documentation |

```bash
curl -F "file=@trace.mseed" http://localhost:8000/api/predict
```

## Command line

```bash
cosmoquake predict trace.mseed              # classify one or more records
cosmoquake predict trace.csv --json         # machine-readable output
cosmoquake catalog data/ -o catalog.csv     # score a directory into a catalog
cosmoquake train                            # retrain and report honest metrics
cosmoquake info                             # what the packaged model knows
cosmoquake serve --port 8080                # run the web service
```

`catalog` emits the column headers the challenge requires for scoring
(`filename`, `time_rel(sec)`, `mq_type`) plus confidence and the raw features.

Detector windows are tunable where the defaults do not suit a record:

```bash
cosmoquake predict trace.mseed --sta 60 --lta 300 --thr-on 3.5 --thr-off 1.2
```

## Python API

```python
from cosmoquake import analyze, load_model

model = load_model()
result = analyze("trace.mseed", model=model)

print(result.label, result.confidence)   # impact_mq 0.983
print(result.detection.arrival_time)     # 1670.1
print(result.features.to_dict())         # the five downlinked numbers
print(result.trustworthy)                # False on non-Apollo data
```

## How it works

**1. Arrival detection (STA/LTA).** A short and a long window slide across the
trace; a quake spikes the short-window average while the long window still
holds background noise. The ratio crossing a threshold marks the onset.

**2. Spectral feature extraction.** Everything after the arrival is
Fourier-transformed. Negative frequencies carry nothing new for a real signal
and are discarded. Five descriptors survive:

| Feature | Meaning |
|---|---|
| `max_velocity` | Peak spectral amplitude — how strong the event was |
| `weighted_velocity` | Amplitude-weighted mean frequency — the spectral centre of mass |
| `mean_velocity` | Mean amplitude across the band |
| `freq_of_max` | Frequency carrying the peak |
| `area` | Area under the frequency-weighted spectrum — a radiated-energy proxy |

**3. Classification.** A random forest sorts the event into impact, deep
moonquake or shallow moonquake. A forest is deliberate: it assumes no
distribution, resists overfitting, selects features on its own, predicts in
logarithmic time, and runs in a power budget a neural network would not fit
into on a rover.

## Honest results

**The classifier reliably identifies impacts. It does not work for deep or
shallow moonquakes, and the training catalog is too small for it to.**

3-fold cross-validation on the 76-event Apollo 12 catalog:

| Metric | Value |
|---|---:|
| Accuracy | 81.6% |
| **Majority-class baseline** | **84.2%** |
| Balanced accuracy | 32.3% |
| `impact_mq` recall | 0.97 |
| `deep_mq` recall | 0.00 |
| `shallow_mq` recall | 0.00 |

Out of fold the model never predicts a rare class, and its rare-class ranking
AUC is ~0.50 — chance. The five features carry no measurable signal separating
deep and shallow moonquakes from impacts here. Rebalancing, log transforms,
depth limits and different estimators were all measured; none found signal that
was not there. Full comparison in the [model card](docs/MODEL_CARD.md).

With 9 deep and 3 shallow events, that ceiling is set by the catalog. The
detection, compression and feature stages hold up independently and are the
parts worth reusing.

Two guardrails keep this visible at runtime:

- Every prediction carries a `trustworthy` flag and, when features land outside
  the training range, a warning explaining why the confidence is not meaningful.
- Applying the lunar model to Mars data (counts/s versus m/s, twelve orders of
  magnitude apart) trips that warning rather than silently returning 98%
  confidence.

## Repository layout

```
src/cosmoquake/        the package
  waveform.py            reading Apollo and InSight records
  detection.py           STA/LTA arrival detection
  features.py            FFT feature extraction
  model.py               training, persistence, out-of-range guard
  pipeline.py            end-to-end analysis
  cli.py                 command-line interface
  web/                   FastAPI service and upload UI
data/moonquakes.csv    76 engineered Apollo 12 events
models/                the trained artifact
tests/                 54 tests
notebooks/             the original hackathon notebooks, kept as a record
Resources/             NASA-provided sample data
docs/MODEL_CARD.md     metrics, limitations, intended use
```

## Deployment

The image is self-contained and honours `$PORT`, so it runs unmodified on
Render, Railway, Fly.io, Cloud Run or any container host.

```bash
docker build -t cosmoquake .
docker run -p 8000:8000 cosmoquake
```

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Listen port |
| `COSMOQUAKE_MODEL` | bundled | Path to a model artifact |
| `COSMOQUAKE_DATASET` | bundled | Path to the training table |
| `COSMOQUAKE_MAX_UPLOAD_MB` | `64` | Upload size limit |

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for platform-specific steps.

## Development

```bash
pip install -e ".[web,dev]"
pytest              # 54 tests
ruff check .
cosmoquake train    # rebuild the model artifact
```

## What changed in v1.0

The original submission was two Jupyter notebooks. The science is unchanged;
the engineering around it was rebuilt, and four defects were fixed along the way:

- **Wrong label mapping.** Inference mapped forest output `2` to `deep_mq` when
  the encoder had assigned `2` to `shallow_mq`, and never emitted `deep_mq` at
  all. Training on string labels removes the hand-written mapping entirely.
- **Sample indices read as seconds.** `trigger_onset` returns sample indices,
  which were compared directly against a time-in-seconds column — off by a
  factor of the sampling rate (20× on InSight data).
- **Hardcoded scaler constants.** The training minima and maxima were pasted
  into the inference code as literals, silently invalidated by any retrain. The
  scaler now travels inside the model artifact.
- **`np.asfarray`.** Removed in NumPy 2.0; the notebooks no longer run on a
  current install.

## Credits

Built by **Team Richter Victors** for NASA Space Apps Challenge 2024.

Data courtesy of NASA's Apollo Passive Seismic Experiment and the InSight
mission (SEIS). Signal processing uses [ObsPy](https://docs.obspy.org/).

## License

[MIT](LICENSE).
