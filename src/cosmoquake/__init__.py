"""Cosmoquake Analyzer: seismic event classification for the Moon and Mars.

The package turns a raw planetary seismogram into five frequency-domain
features and classifies the event behind it, reducing gigabytes of telemetry
to a handful of numbers that fit in a low-bandwidth downlink budget.
"""

from cosmoquake.detection import DetectionResult, detect_arrival
from cosmoquake.features import FEATURE_NAMES, FeatureVector, extract_features
from cosmoquake.model import CosmoquakeModel, load_model, train_model
from cosmoquake.pipeline import Prediction, analyze
from cosmoquake.waveform import Waveform, read_waveform

__all__ = [
    "FEATURE_NAMES",
    "CosmoquakeModel",
    "DetectionResult",
    "FeatureVector",
    "Prediction",
    "Waveform",
    "analyze",
    "detect_arrival",
    "extract_features",
    "load_model",
    "read_waveform",
    "train_model",
]

__version__ = "1.0.0"
