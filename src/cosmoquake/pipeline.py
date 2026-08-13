"""End-to-end analysis: raw trace in, classified event out.

    read -> STA/LTA arrival -> trim -> FFT features -> forest -> label

Each stage lives in its own module; this one wires them together and reports
the intermediate values, because on a science instrument the arrival time and
the spectrum matter as much as the final label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cosmoquake.detection import (
    DEFAULT_LTA_SECONDS,
    DEFAULT_STA_SECONDS,
    DEFAULT_THRESHOLD_OFF,
    DEFAULT_THRESHOLD_ON,
    DetectionResult,
    detect_arrival,
)
from cosmoquake.features import FeatureVector, extract_features
from cosmoquake.model import CosmoquakeModel, describe_class, load_model
from cosmoquake.waveform import Waveform, read_waveform

#: Bytes on the wire for one classified event: five float64 features plus
#: the arrival time. Used to report the compression ratio.
DOWNLINK_BYTES = 6 * 8


@dataclass(frozen=True)
class Prediction:
    """The full result of analysing one trace.

    Attributes:
        source: Name of the analysed file.
        label: Predicted event class.
        description: Human-readable form of ``label``.
        confidence: Probability the model assigned to ``label``.
        probabilities: Probability for every class, highest first.
        detection: Where the arrival was found.
        features: The five extracted descriptors.
        n_samples: Samples in the original record.
        duration: Length of the original record in seconds.
        sampling_rate: Samples per second in the original record.
        out_of_range: Features outside the model's training range, mapped to
            how many range-widths beyond the edge they sit. Empty when the
            record resembles the training data.
    """

    source: str
    label: str
    description: str
    confidence: float
    probabilities: dict[str, float]
    detection: DetectionResult
    features: FeatureVector
    n_samples: int
    duration: float
    sampling_rate: float
    out_of_range: dict[str, float] = field(default_factory=dict)

    @property
    def trustworthy(self) -> bool:
        """Whether the record sits within the range the model was trained on."""
        return not self.out_of_range

    @property
    def warning(self) -> str | None:
        """A human-readable caveat, or ``None`` when the result is in range."""
        if self.trustworthy:
            return None
        worst = max(self.out_of_range.items(), key=lambda pair: pair[1])
        return (
            f"{len(self.out_of_range)} of {len(self.features.to_dict())} features fall "
            f"outside the training range (worst: {worst[0]}, {worst[1]:,.1f}x the range "
            f"beyond its edge). The model was trained on Apollo lunar data in m/s; "
            f"confidence is not meaningful for records in other units or from other "
            f"bodies."
        )

    @property
    def raw_bytes(self) -> int:
        """Approximate size of the record as float64 samples."""
        return self.n_samples * 8

    @property
    def compression_ratio(self) -> float:
        """How many times smaller the downlinked summary is than the record."""
        return self.raw_bytes / DOWNLINK_BYTES

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the whole result."""
        return {
            "source": self.source,
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "detection": self.detection.to_dict(),
            "features": self.features.to_dict(),
            "trustworthy": self.trustworthy,
            "out_of_range": self.out_of_range,
            "warning": self.warning,
            "record": {
                "n_samples": self.n_samples,
                "duration_seconds": self.duration,
                "sampling_rate": self.sampling_rate,
                "raw_bytes": self.raw_bytes,
                "downlink_bytes": DOWNLINK_BYTES,
                "compression_ratio": self.compression_ratio,
            },
        }


def analyze_waveform(
    waveform: Waveform,
    model: CosmoquakeModel,
    sta_seconds: float = DEFAULT_STA_SECONDS,
    lta_seconds: float = DEFAULT_LTA_SECONDS,
    threshold_on: float = DEFAULT_THRESHOLD_ON,
    threshold_off: float = DEFAULT_THRESHOLD_OFF,
    detection_waveform: Waveform | None = None,
) -> Prediction:
    """Detect, describe and classify a single trace.

    Args:
        waveform: Trace to extract features from.
        model: Classifier to apply.
        sta_seconds: Short-term STA/LTA window.
        lta_seconds: Long-term STA/LTA window.
        threshold_on: Trigger-on ratio.
        threshold_off: Trigger-off ratio.
        detection_waveform: Optional separate trace to run the trigger on,
            for the case where miniSEED and CSV exports of the same event are
            both available. Defaults to ``waveform``.
    """
    detection = detect_arrival(
        detection_waveform if detection_waveform is not None else waveform,
        sta_seconds=sta_seconds,
        lta_seconds=lta_seconds,
        threshold_on=threshold_on,
        threshold_off=threshold_off,
    )

    event = waveform.after(detection.arrival_time) if detection.triggered else waveform
    features = extract_features(event)
    probabilities = model.predict_proba(features)
    label = next(iter(probabilities))

    return Prediction(
        source=Path(waveform.source).name if waveform.source else "<memory>",
        label=label,
        description=describe_class(label),
        confidence=probabilities[label],
        probabilities=probabilities,
        detection=detection,
        features=features,
        n_samples=len(waveform),
        duration=waveform.duration,
        sampling_rate=waveform.sampling_rate,
        out_of_range=model.out_of_range_features(features),
    )


def analyze(
    path: str | Path,
    model: CosmoquakeModel | None = None,
    detection_path: str | Path | None = None,
    **kwargs: Any,
) -> Prediction:
    """Analyse a seismic file.

    Args:
        path: A ``.csv`` or ``.mseed`` record.
        model: A loaded model; the packaged one is used when omitted.
        detection_path: Optional companion file to run the trigger on.
        **kwargs: STA/LTA settings forwarded to :func:`analyze_waveform`.

    Returns:
        The :class:`Prediction` for the event in the file.
    """
    model = model if model is not None else load_model()
    waveform = read_waveform(path)
    detection_waveform = read_waveform(detection_path) if detection_path else None
    return analyze_waveform(
        waveform, model, detection_waveform=detection_waveform, **kwargs
    )
