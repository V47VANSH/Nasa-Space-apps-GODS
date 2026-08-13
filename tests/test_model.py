"""Training, persistence and the out-of-range guard."""

from __future__ import annotations

import pytest

from cosmoquake.features import FeatureVector, extract_features
from cosmoquake.model import ModelError, load_model, train_model


def test_trained_model_knows_the_catalog_classes(model):
    assert set(model.classes) == {"impact_mq", "deep_mq", "shallow_mq"}


def test_probabilities_sum_to_one(model, synthetic_quake):
    probabilities = model.predict_proba(extract_features(synthetic_quake))
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert set(probabilities) == set(model.classes)


def test_probabilities_are_ranked_highest_first(model, synthetic_quake):
    scores = list(model.predict_proba(extract_features(synthetic_quake)).values())
    assert scores == sorted(scores, reverse=True)


def test_predict_agrees_with_the_top_probability(model, synthetic_quake):
    features = extract_features(synthetic_quake)
    top = next(iter(model.predict_proba(features)))
    assert model.predict(features) == top


def test_labels_are_names_not_encoded_integers(model, synthetic_quake):
    """Regression: an earlier version mapped forest outputs to the wrong names.

    Training on string labels removes the hand-written integer mapping that
    silently swapped deep and shallow events.
    """
    assert model.predict(extract_features(synthetic_quake)) in {
        "impact_mq",
        "deep_mq",
        "shallow_mq",
    }


def test_round_trip_through_disk_preserves_predictions(model, synthetic_quake, tmp_path):
    path = model.save(tmp_path / "model.joblib")
    reloaded = load_model(path)

    features = extract_features(synthetic_quake)
    assert reloaded.predict(features) == model.predict(features)
    assert reloaded.classes == model.classes
    assert reloaded.metadata == model.metadata


def test_scaler_travels_with_the_model(model):
    """Regression: scaling constants were once pasted into the predict code."""
    ranges = model.training_range()
    assert set(ranges) == set(model.feature_names)
    assert all(low < high for low, high in ranges.values())


def test_missing_artefact_explains_how_to_fix_it(tmp_path):
    with pytest.raises(ModelError, match="cosmoquake train"):
        load_model(tmp_path / "absent.joblib")


def test_bare_estimator_is_rejected(tmp_path):
    """A legacy joblib holding only an estimator must not load silently."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    path = tmp_path / "legacy.joblib"
    joblib.dump(RandomForestClassifier(), path)
    with pytest.raises(ModelError, match="not a Cosmoquake artefact"):
        load_model(path)


def test_newer_artefact_version_is_rejected(tmp_path, model):
    import joblib

    path = tmp_path / "future.joblib"
    joblib.dump(
        {
            "artifact_version": 999,
            "pipeline": model.pipeline,
            "feature_names": list(model.feature_names),
            "metadata": {},
        },
        path,
    )
    with pytest.raises(ModelError, match="newer release"):
        load_model(path)


def test_in_range_features_raise_no_warning(model):
    """A vector drawn from the middle of the training range is trusted."""
    midpoints = {
        name: (low + high) / 2 for name, (low, high) in model.training_range().items()
    }
    assert model.out_of_range_features(FeatureVector(**midpoints)) == {}


def test_out_of_range_features_are_flagged(model):
    """InSight counts/s sit far outside Apollo m/s and must be caught."""
    huge = FeatureVector(
        max_velocity=9e3,
        weighted_velocity=1e3,
        mean_velocity=1e3,
        freq_of_max=2.4,
        area=1e8,
    )
    flagged = model.out_of_range_features(huge)
    assert "area" in flagged
    assert flagged["area"] > 1.0


def test_metrics_report_the_majority_baseline():
    """Accuracy on this catalog is meaningless without its baseline."""
    trained = train_model(evaluate=True)
    scores = trained.metadata["cross_validation"]
    assert scores["majority_baseline"] == pytest.approx(64 / 76, rel=0.01)
    assert 0.0 <= scores["balanced_accuracy"] <= 1.0
    assert set(scores["per_class"]) == {"impact_mq", "deep_mq", "shallow_mq"}


def test_feature_importances_cover_every_feature(model):
    importances = model.feature_importances()
    assert set(importances) == set(model.feature_names)
    assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)


def test_training_is_reproducible():
    """The same seed must rebuild the same model."""
    first = train_model(random_state=7, n_estimators=50, evaluate=False)
    second = train_model(random_state=7, n_estimators=50, evaluate=False)
    sample = FeatureVector(
        max_velocity=1e-11,
        weighted_velocity=1e-13,
        mean_velocity=1e-13,
        freq_of_max=0.5,
        area=1e-8,
    )
    assert first.predict(sample) == second.predict(sample)
    assert first.predict_proba(sample) == second.predict_proba(sample)
