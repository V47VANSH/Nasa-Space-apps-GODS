"""Frequency-domain feature extraction.

This is where the compression happens. A one-hour InSight record is roughly
72 MB of samples; the five numbers produced here describe the same event in
40 bytes, which is what makes the results downlinkable from a rover.

The arithmetic deliberately matches the original hackathon notebooks so
models trained before this refactor stay valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.fft import fft, fftfreq

from cosmoquake.waveform import Waveform

#: Feature order expected by every model in this project. Changing this
#: order invalidates saved models, so append rather than reorder.
FEATURE_NAMES = (
    "max_velocity",
    "weighted_velocity",
    "mean_velocity",
    "freq_of_max",
    "area",
)


@dataclass(frozen=True)
class FeatureVector:
    """The five descriptors of a single seismic event.

    Attributes:
        max_velocity: Peak spectral amplitude; how strong the event was.
        weighted_velocity: Amplitude-weighted mean frequency, a centre of
            spectral mass that separates sharp impacts from slow deep events.
        mean_velocity: Mean spectral amplitude across the band.
        freq_of_max: Frequency carrying the peak amplitude.
        area: Area under the frequency-weighted spectrum; total radiated
            energy proxy.
    """

    max_velocity: float
    weighted_velocity: float
    mean_velocity: float
    freq_of_max: float
    area: float

    def to_array(self) -> np.ndarray:
        """Return the features as a 1-D array in :data:`FEATURE_NAMES` order."""
        return np.array([getattr(self, name) for name in FEATURE_NAMES], dtype=float)

    def to_row(self) -> np.ndarray:
        """Return the features shaped ``(1, 5)`` for scikit-learn estimators."""
        return self.to_array().reshape(1, -1)

    def to_dict(self) -> dict[str, float]:
        """Return the features as a plain JSON-serialisable mapping."""
        return asdict(self)


def spectrum(waveform: Waveform) -> tuple[np.ndarray, np.ndarray]:
    """Return the single-sided amplitude spectrum of a trace.

    Negative frequencies carry no extra information for a real-valued signal,
    so only the positive half is kept and amplitudes are doubled to conserve
    energy.

    Returns:
        A ``(frequencies, amplitudes)`` pair, both of length ``N // 2``.
    """
    amplitude = np.asarray(waveform.velocity, dtype=float)
    n_samples = amplitude.size
    half = n_samples // 2
    if half < 1:
        raise ValueError("waveform is too short to transform")

    amplitudes = 2.0 / n_samples * np.abs(fft(amplitude)[:half])
    frequencies = fftfreq(n_samples, waveform.sample_spacing)[:half]
    return frequencies, amplitudes


def extract_features(waveform: Waveform) -> FeatureVector:
    """Reduce a trace to its :class:`FeatureVector`.

    Args:
        waveform: The trace, typically already trimmed to start at the
            detected arrival time.
    """
    frequencies, amplitudes = spectrum(waveform)

    frequency_sum = float(np.sum(frequencies))
    weighted_area = float(np.sum(amplitudes * frequencies))
    # The DC-only edge case: a flat trace has every frequency at zero, and
    # the weighted mean is undefined rather than infinite.
    weighted_velocity = weighted_area / frequency_sum if frequency_sum else 0.0

    return FeatureVector(
        max_velocity=float(np.max(amplitudes)),
        weighted_velocity=weighted_velocity,
        mean_velocity=float(np.mean(amplitudes)),
        freq_of_max=float(frequencies[int(np.argmax(amplitudes))]),
        area=weighted_area,
    )
