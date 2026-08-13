# Model card — Cosmoquake event classifier

## Overview

| | |
|---|---|
| **Task** | Classify a lunar seismic event as impact, deep moonquake or shallow moonquake |
| **Architecture** | `MinMaxScaler` → `RandomForestClassifier` (300 trees, `class_weight="balanced_subsample"`) |
| **Input** | Five frequency-domain features extracted from the post-arrival waveform |
| **Training data** | 76 events from the Apollo 12 Grade-A catalog (`data/moonquakes.csv`) |
| **Artifact** | `models/cosmoquake_rf.joblib` |
| **Reproduce** | `cosmoquake train` (seeded, `random_state=42`) |

## Training data

Derived from the NASA Space Apps 2024 "Seismic Detection Across the Solar
System" challenge data. Each row is one catalogued Apollo 12 event, reduced to
five features from the FFT of the waveform following the detected arrival.

| Class | Count | Share |
|---|---:|---:|
| `impact_mq` — meteoroid impact | 64 | 84.2% |
| `deep_mq` — deep moonquake | 9 | 11.8% |
| `shallow_mq` — shallow moonquake | 3 | 3.9% |

## Performance

3-fold stratified cross-validation — the fold count is capped by the three
shallow events. Reproduce with `cosmoquake info`.

| Metric | Value |
|---|---:|
| Accuracy | 81.6% |
| **Majority-class baseline** | **84.2%** |
| Balanced accuracy | 32.3% |
| Macro F1 | 0.300 |

Per class:

| Class | Precision | Recall | Support | One-vs-rest AUC |
|---|---:|---:|---:|---:|
| `impact_mq` | 0.84 | 0.97 | 64 | 0.32 |
| `deep_mq` | 0.00 | 0.00 | 9 | 0.56 |
| `shallow_mq` | 0.00 | 0.00 | 3 | 0.50 |

### What these numbers mean

**The classifier does not work for the rare classes, and accuracy hides it.**
Out of fold it never once predicts `deep_mq` or `shallow_mq`, and its overall
accuracy is *below* the 84.2% you get by always guessing `impact_mq`.

The AUC column rules out the most hopeful reading. If the model were ranking
rare events correctly but simply thresholding badly, rare-class AUC would be
high even with zero recall. At 0.50 and 0.56, it is not: the five features
carry essentially **no signal** separating deep and shallow moonquakes from
impacts in this catalog.

This was tested, not assumed. Alternatives measured under the same 3-fold split:

| Configuration | Accuracy | Balanced acc. | `deep_mq` recall | `shallow_mq` recall |
|---|---:|---:|---:|---:|
| No class weighting | 84.2% | 33.3% | 0.00 | 0.00 |
| `balanced_subsample` (shipped) | 81.6% | 32.3% | 0.00 | 0.00 |
| `balanced` (original notebook) | 64.5% | 25.5% | 0.00 | 0.00 |
| Depth-3 forest, `balanced` | 51.3% | 26.7% | 0.22 | 0.00 |
| Logistic regression, `balanced` | 27.6% | 40.6% | 0.67 | 0.33 |
| log₁₀ feature transform (any of the above) | no change | | | |

Rebalancing trades accuracy for rare-class recall without ever finding real
signal. Logistic regression recovers some rare-class recall only by predicting
those classes constantly, at 27.6% accuracy — worse than useless in a survey
where impacts genuinely dominate.

The shipped default (`balanced_subsample`) keeps probability mass on the rare
classes for triage purposes while staying near baseline accuracy. Change it
with `cosmoquake train --class-weight none`.

### Why the data cannot support more

Three shallow moonquakes is not a training set. Even a perfect feature
representation could not be validated at that support, and a single
misclassification moves recall by 33 percentage points. The honest ceiling here
is set by the catalog, not by the model family.

## Intended use

**Appropriate.** Demonstrating the detect-compress-classify pipeline;
identifying impact events, which the model does handle (0.97 recall); a
teaching example of STA/LTA plus spectral feature engineering; a starting point
for work with a larger catalog.

**Not appropriate.** Any scientific claim about deep or shallow moonquakes;
operational deployment on a real mission; use on non-lunar data (see below);
any decision where a false negative on a rare event carries a cost.

## Limitations

- **Lunar Apollo data only.** InSight records velocity in counts/s and Apollo
  in m/s — about twelve orders of magnitude apart. Applying this model to Mars
  data produces confident, meaningless output. The pipeline detects this and
  attaches a warning (`trustworthy: false`); it does not refuse the prediction,
  because the *detection and feature stages* remain valid on Mars data even
  when the classifier is not.
- **Arrival dependence.** Features are computed on everything after the STA/LTA
  arrival. A missed or mistimed trigger changes them substantially.
- **No epicentral distance.** True magnitude cannot be recovered without the
  source-to-station distance, which the catalog does not provide.
- **Single station.** Apollo 12 only; no cross-station validation.

## What is trustworthy in this project

The signal-processing stages hold up independently of the classifier:

- STA/LTA arrival detection works on both Apollo and InSight records.
- FFT feature extraction is deterministic and unit-tested against synthetic
  tones with known spectra.
- The compression claim is real and measurable: a one-hour InSight record is
  reduced from ~576 KB of float64 samples to 48 bytes, a factor of ~12,000.

The classifier is the weak link, and only because of catalog size.

## Ethical and safety notes

No personal data. The failure mode of consequence is scientific
over-confidence: reporting rare-event classifications that the evidence does
not support. The metrics above, the out-of-range warning and the `trustworthy`
flag in every API response exist to make that failure visible rather than
silent.
