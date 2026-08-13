"""Training, saving and loading the event classifier.

The model is a two-step scikit-learn pipeline: min-max scaling followed by a
random forest. Min-max is deliberate — the features span eleven orders of
magnitude and are nowhere near normal, so standardising them would impose a
distribution the data does not have.

The scaler travels *inside* the saved artefact. An earlier version of this
project pasted the training minima and maxima into the prediction code as
literals, which silently rots the moment the model is retrained.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from cosmoquake.features import FEATURE_NAMES, FeatureVector


def _resolve_default(env_var: str, *relative: str) -> Path:
    """Find a bundled file across editable, installed and container layouts.

    Searched in order: the environment variable, then the same path relative
    to each ancestor of this module, then the current working directory. An
    editable install resolves at the repository root; a wheel installed into
    ``site-packages`` and run from a container's ``/app`` resolves via the
    working directory instead. The last candidate is returned unchanged when
    nothing exists, so callers report a sensible path in their error message.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()

    candidates = [parent.joinpath(*relative) for parent in Path(__file__).resolve().parents]
    candidates.append(Path.cwd().joinpath(*relative))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[2] if len(candidates) > 2 else candidates[-1]


#: Default location of the trained artefact. Override with ``COSMOQUAKE_MODEL``.
DEFAULT_MODEL_PATH = _resolve_default("COSMOQUAKE_MODEL", "models", "cosmoquake_rf.joblib")

#: Default training table. Override with ``COSMOQUAKE_DATASET``.
DEFAULT_DATASET_PATH = _resolve_default("COSMOQUAKE_DATASET", "data", "moonquakes.csv")

#: Bumped whenever the artefact layout changes in a way loaders must notice.
ARTIFACT_VERSION = 2

#: Human-readable names for the classes in the Apollo 12 Grade-A catalog.
CLASS_DESCRIPTIONS = {
    "impact_mq": "Meteoroid impact",
    "deep_mq": "Deep moonquake (tidal, ~700-1200 km)",
    "shallow_mq": "Shallow moonquake (rare, high-frequency)",
}


class ModelError(RuntimeError):
    """Raised when a model artefact is missing or incompatible."""


@dataclass
class CosmoquakeModel:
    """A trained classifier together with everything needed to interpret it.

    Attributes:
        pipeline: Fitted scaler-plus-forest pipeline.
        feature_names: Feature order the pipeline was fitted on.
        metadata: Training provenance and cross-validated scores.
    """

    pipeline: Pipeline
    feature_names: tuple[str, ...] = FEATURE_NAMES
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def classes(self) -> tuple[str, ...]:
        """Class labels the model can emit, in the order it reports scores."""
        return tuple(str(label) for label in self.pipeline.classes_)

    def _as_frame(self, features: FeatureVector) -> pd.DataFrame:
        """Wrap a feature vector in a named frame the pipeline expects."""
        return pd.DataFrame(features.to_row(), columns=list(self.feature_names))

    def predict(self, features: FeatureVector) -> str:
        """Return the most likely class for one feature vector."""
        return str(self.pipeline.predict(self._as_frame(features))[0])

    def predict_proba(self, features: FeatureVector) -> dict[str, float]:
        """Return per-class probabilities, highest first."""
        scores = self.pipeline.predict_proba(self._as_frame(features))[0]
        ranked = sorted(
            zip(self.classes, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return {label: float(score) for label, score in ranked}

    def training_range(self) -> dict[str, tuple[float, float]]:
        """Return the min and max of each feature seen during training."""
        scaler = self.pipeline.named_steps["scaler"]
        return {
            name: (float(low), float(high))
            for name, low, high in zip(
                self.feature_names, scaler.data_min_, scaler.data_max_, strict=True
            )
        }

    def out_of_range_features(self, features: FeatureVector) -> dict[str, float]:
        """Return features lying outside the training range, and by how far.

        A random forest extrapolates by saturating: anything above the
        largest value it ever saw falls into the same leaf as that value, and
        the reported probability stays confident while being meaningless.
        Apollo records velocity in m/s and InSight in counts/s, roughly
        twelve orders of magnitude apart, so a model trained on one and
        applied to the other lands far outside its range without any
        intrinsic signal that something is wrong. Callers use this to warn.

        Returns:
            A mapping of feature name to the number of training-range widths
            the value sits beyond the nearest edge. Empty when in range.
        """
        deviations: dict[str, float] = {}
        for name, value in features.to_dict().items():
            low, high = self.training_range()[name]
            width = high - low
            if width <= 0:
                continue
            if value < low:
                deviations[name] = float((low - value) / width)
            elif value > high:
                deviations[name] = float((value - high) / width)
        return deviations

    def feature_importances(self) -> dict[str, float]:
        """Return each feature's importance in the forest, highest first."""
        forest = self.pipeline.named_steps["classifier"]
        pairs = sorted(
            zip(self.feature_names, forest.feature_importances_, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return {name: float(value) for name, value in pairs}

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        """Write the model to ``path``, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_version": ARTIFACT_VERSION,
                "pipeline": self.pipeline,
                "feature_names": list(self.feature_names),
                "metadata": self.metadata,
            },
            path,
        )
        return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> CosmoquakeModel:
    """Load a model artefact written by :meth:`CosmoquakeModel.save`.

    Raises:
        ModelError: If the file is missing or was written by an
            incompatible version of this package.
    """
    path = Path(path)
    if not path.exists():
        raise ModelError(
            f"no model at {path}. Train one with `cosmoquake train`, "
            f"or point --model at an existing artefact."
        )

    payload = joblib.load(path)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ModelError(
            f"{path} is not a Cosmoquake artefact (it may be a bare estimator "
            f"from an older release). Retrain with `cosmoquake train`."
        )

    version = payload.get("artifact_version", 0)
    if version > ARTIFACT_VERSION:
        raise ModelError(
            f"{path} was written by a newer release (artifact version "
            f"{version} > {ARTIFACT_VERSION}). Upgrade the cosmoquake package."
        )

    return CosmoquakeModel(
        pipeline=payload["pipeline"],
        feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
        metadata=payload.get("metadata", {}),
    )


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature table used for training.

    Args:
        path: CSV with one column per entry in :data:`FEATURE_NAMES` plus a
            ``label`` column.

    Returns:
        A ``(features, labels)`` pair.
    """
    path = Path(path)
    if not path.exists():
        raise ModelError(f"training data not found at {path}")

    frame = pd.read_csv(path)
    missing = [name for name in (*FEATURE_NAMES, "label") if name not in frame.columns]
    if missing:
        raise ModelError(f"{path} is missing required columns: {missing}")

    frame = frame[[*FEATURE_NAMES, "label"]].dropna()
    return frame[list(FEATURE_NAMES)], frame["label"].astype(str)


def _cross_validated_report(
    pipeline: Pipeline, features: pd.DataFrame, labels: pd.Series, seed: int
) -> dict[str, Any]:
    """Score the pipeline out-of-fold, or explain why it could not be scored.

    The Apollo catalog has three shallow moonquakes in seventy-six events, so
    the number of folds is capped by the rarest class. Reporting a plain
    training-set accuracy here would be meaningless for a forest.

    Accuracy alone would flatter this model badly: 84% of the catalog is one
    class, so a constant predictor scores 84%. The majority baseline, the
    balanced accuracy and the per-class ranking AUCs are all reported so the
    headline number cannot be read out of context.
    """
    counts = labels.value_counts()
    majority_baseline = float(counts.max() / counts.sum())
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        return {
            "cross_validation": None,
            "majority_baseline": majority_baseline,
            "note": (
                f"skipped: rarest class has {int(counts.min())} sample(s), "
                f"which cannot be split across folds"
            ),
        }

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predicted = cross_val_predict(pipeline, features, labels, cv=splitter)
    probabilities = cross_val_predict(
        pipeline, features, labels, cv=splitter, method="predict_proba"
    )
    report = classification_report(labels, predicted, output_dict=True, zero_division=0)

    # One-vs-rest AUC per class: does the model at least rank true events of
    # this class above the rest, even when it never predicts the class
    # outright? An AUC near 0.5 means the features carry no signal for it.
    class_order = sorted(labels.unique())
    ranking_auc: dict[str, float | None] = {}
    for index, label in enumerate(class_order):
        actual = (labels == label).astype(int)
        try:
            ranking_auc[label] = float(roc_auc_score(actual, probabilities[:, index]))
        except ValueError:  # a class absent from some fold
            ranking_auc[label] = None

    return {
        "cross_validation": {
            "n_splits": n_splits,
            "accuracy": float(report["accuracy"]),
            "majority_baseline": majority_baseline,
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_class": {
                label: {
                    "precision": float(scores["precision"]),
                    "recall": float(scores["recall"]),
                    "f1": float(scores["f1-score"]),
                    "support": int(scores["support"]),
                    "ranking_auc": ranking_auc.get(label),
                }
                for label, scores in report.items()
                if label in set(labels.unique())
            },
        }
    }


def train_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    n_estimators: int = 300,
    random_state: int = 42,
    evaluate: bool = True,
    class_weight: str | None = "balanced_subsample",
) -> CosmoquakeModel:
    """Fit the classifier on the feature table.

    Args:
        dataset_path: CSV of engineered features and labels.
        n_estimators: Trees in the forest.
        random_state: Seed, so a rebuild reproduces the shipped artefact.
        evaluate: Whether to compute cross-validated scores, which roughly
            doubles training time.
        class_weight: Rebalancing strategy passed to the forest. The default
            reweights within each bootstrap sample. Plain ``"balanced"``,
            used by the original notebook, costs about 20 points of accuracy
            on this catalog without recovering any rare-class recall.

    Returns:
        The fitted :class:`CosmoquakeModel`, not yet written to disk.
    """
    features, labels = load_dataset(dataset_path)

    pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight=class_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    metadata: dict[str, Any] = {
        "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "dataset": str(Path(dataset_path).name),
        "n_samples": len(labels),
        "class_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "n_estimators": n_estimators,
        "random_state": random_state,
        "class_weight": class_weight,
    }
    if evaluate:
        metadata.update(_cross_validated_report(pipeline, features, labels, random_state))

    pipeline.fit(features, labels)
    metadata["feature_importances"] = {
        name: float(value)
        for name, value in zip(
            FEATURE_NAMES,
            pipeline.named_steps["classifier"].feature_importances_,
            strict=True,
        )
    }

    return CosmoquakeModel(pipeline=pipeline, metadata=metadata)


def describe_class(label: str) -> str:
    """Return a human-readable description for a class label."""
    return CLASS_DESCRIPTIONS.get(label, label)


def features_to_frame(vectors: list[FeatureVector]) -> pd.DataFrame:
    """Stack feature vectors into a dataframe with named columns."""
    return pd.DataFrame(
        np.vstack([vector.to_array() for vector in vectors]), columns=list(FEATURE_NAMES)
    )
