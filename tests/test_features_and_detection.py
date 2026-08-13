"""Feature extraction and STA/LTA detection."""

from __future__ import annotations

import numpy as np
import pytest

from cosmoquake.detection import detect_arrival
from cosmoquake.features import FEATURE_NAMES, extract_features, spectrum
from cosmoquake.waveform import Waveform


def sine_wave(frequency: float, rate: float = 50.0, duration: float = 60.0) -> Waveform:
    """A pure tone, whose spectrum has a known single peak."""
    time = np.arange(0.0, duration, 1.0 / rate)
    return Waveform(time=time, velocity=np.sin(2 * np.pi * frequency * time))


def test_spectrum_finds_a_known_tone():
    frequencies, amplitudes = spectrum(sine_wave(3.0))
    assert frequencies[int(np.argmax(amplitudes))] == pytest.approx(3.0, abs=0.05)


def test_spectrum_keeps_only_positive_frequencies():
    frequencies, _ = spectrum(sine_wave(3.0))
    assert (frequencies >= 0).all()


def test_features_are_finite_and_ordered(synthetic_quake):
    features = extract_features(synthetic_quake)
    array = features.to_array()
    assert array.shape == (5,)
    assert np.isfinite(array).all()
    assert list(features.to_dict()) == list(FEATURE_NAMES)


def test_amplitude_scales_the_velocity_features():
    """Doubling the trace doubles the amplitude features, not the frequency."""
    base = extract_features(sine_wave(3.0))
    louder_wave = sine_wave(3.0)
    louder = extract_features(
        Waveform(louder_wave.time, louder_wave.velocity * 2.0, "louder")
    )
    assert louder.max_velocity == pytest.approx(base.max_velocity * 2.0, rel=1e-6)
    assert louder.freq_of_max == pytest.approx(base.freq_of_max)


def test_flat_trace_does_not_divide_by_zero(flat_trace):
    """A dead channel must yield zeros, not NaN."""
    features = extract_features(flat_trace)
    assert np.isfinite(features.to_array()).all()
    assert features.max_velocity == 0.0


def test_detects_a_synthetic_arrival(synthetic_quake):
    """The trigger finds the burst that was planted at t=600 s."""
    result = detect_arrival(synthetic_quake, sta_seconds=20, lta_seconds=120)
    assert result.triggered
    assert result.arrival_time == pytest.approx(600.0, abs=30.0)


def test_quiet_trace_reports_no_trigger(flat_trace):
    result = detect_arrival(flat_trace, sta_seconds=10, lta_seconds=60)
    assert not result.triggered
    assert result.n_triggers == 0
    assert result.arrival_time == flat_trace.time[0]


def test_arrival_is_in_seconds_not_sample_indices(mars_csv):
    """Regression: trigger_onset returns indices, which must be converted.

    At InSight's 20 Hz an unconverted index reads twenty times too large and
    lands past the end of a one-hour record.
    """
    from cosmoquake.waveform import read_waveform

    waveform = read_waveform(mars_csv)
    result = detect_arrival(waveform)
    assert 0.0 <= result.arrival_time <= waveform.duration


def test_short_record_does_not_crash_the_detector():
    """Windows longer than the record are clamped rather than raising."""
    time = np.arange(0.0, 30.0, 0.1)
    waveform = Waveform(time=time, velocity=np.random.default_rng(1).normal(0, 1, time.size))
    result = detect_arrival(waveform, sta_seconds=120, lta_seconds=600)
    assert isinstance(result.triggered, bool)
