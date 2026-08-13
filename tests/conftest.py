"""Shared fixtures.

Tests run against a model trained on the shipped catalog, and against the
real InSight records in ``Resources/`` when they are present. The data is
large enough that a shallow clone may omit it, so those tests skip rather
than fail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cosmoquake.model import DEFAULT_DATASET_PATH, train_model
from cosmoquake.waveform import Waveform

REPO_ROOT = Path(__file__).resolve().parents[1]
MARS_TEST_DIR = REPO_ROOT / "Resources" / "Data" / "mars" / "test" / "data"


@pytest.fixture(scope="session")
def model():
    """A model trained once for the whole session; training is not free."""
    if not DEFAULT_DATASET_PATH.exists():
        pytest.skip(f"training data missing at {DEFAULT_DATASET_PATH}")
    return train_model(evaluate=False)


@pytest.fixture
def synthetic_quake() -> Waveform:
    """A quiet trace with a decaying burst starting at t=600 s.

    Built rather than loaded so the arrival time is known exactly and the
    detector can be checked against ground truth.
    """
    rate = 10.0
    duration = 1200.0
    time = np.arange(0.0, duration, 1.0 / rate)

    rng = np.random.default_rng(0)
    velocity = rng.normal(0.0, 1e-10, time.size)

    onset = 600.0
    burst = time >= onset
    envelope = np.exp(-(time[burst] - onset) / 60.0)
    velocity[burst] += 5e-9 * envelope * np.sin(2 * np.pi * 1.5 * (time[burst] - onset))

    return Waveform(time=time, velocity=velocity, source="synthetic.csv")


@pytest.fixture
def flat_trace() -> Waveform:
    """A trace with no event in it at all."""
    time = np.arange(0.0, 600.0, 0.1)
    return Waveform(time=time, velocity=np.zeros_like(time), source="flat.csv")


@pytest.fixture
def mars_csv() -> Path:
    """A real InSight record, skipped when the sample data is absent."""
    candidates = sorted(MARS_TEST_DIR.glob("*.csv"))
    if not candidates:
        pytest.skip("InSight sample data not present in this checkout")
    return candidates[0]


@pytest.fixture
def mars_mseed() -> Path:
    """The miniSEED counterpart of a real InSight record."""
    candidates = sorted(MARS_TEST_DIR.glob("*.mseed"))
    if not candidates:
        pytest.skip("InSight sample data not present in this checkout")
    return candidates[0]
