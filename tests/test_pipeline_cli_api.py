"""End-to-end analysis, the CLI and the HTTP service."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from cosmoquake.cli import iter_records, main
from cosmoquake.pipeline import analyze, analyze_waveform
from cosmoquake.web.app import app

# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def test_analyze_produces_a_complete_result(model, mars_csv):
    result = analyze(mars_csv, model=model)
    assert result.label in model.classes
    assert 0.0 <= result.confidence <= 1.0
    assert result.n_samples > 0
    assert result.compression_ratio > 1.0


def test_result_serialises_to_json(model, mars_csv):
    payload = analyze(mars_csv, model=model).to_dict()
    json.dumps(payload)  # must not raise on numpy scalars
    assert {"label", "features", "detection", "record", "warning"} <= set(payload)


def test_mseed_and_csv_of_one_event_classify_alike(model, mars_mseed):
    partner = mars_mseed.with_suffix(".csv")
    if not partner.exists():
        pytest.skip("no CSV counterpart")
    assert analyze(mars_mseed, model=model).label == analyze(partner, model=model).label


def test_insight_record_is_flagged_out_of_range(model, mars_csv):
    """Mars counts/s cannot be judged by a model trained on Apollo m/s."""
    result = analyze(mars_csv, model=model)
    assert not result.trustworthy
    assert result.warning and "training range" in result.warning


def test_untriggered_trace_still_classifies(model, flat_trace):
    result = analyze_waveform(flat_trace, model)
    assert not result.detection.triggered
    assert result.label in model.classes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_predict_prints_a_report(mars_csv, capsys):
    assert main(["predict", str(mars_csv)]) == 0
    output = capsys.readouterr().out
    assert "confidence" in output
    assert "features" in output


def test_cli_predict_json_is_parseable(mars_csv, capsys):
    assert main(["predict", str(mars_csv), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["label"]


def test_cli_train_writes_an_artefact(tmp_path, capsys):
    out = tmp_path / "m.joblib"
    assert main(["train", "--out", str(out), "--trees", "20", "--no-eval"]) == 0
    assert out.exists()
    assert "Saved model" in capsys.readouterr().out


def test_cli_info_reports_the_baseline(capsys):
    assert main(["info"]) == 0
    assert "majority baseline" in capsys.readouterr().out


def test_cli_catalog_writes_the_challenge_columns(tmp_path, mars_csv, capsys):
    out = tmp_path / "catalog.csv"
    assert main(["catalog", str(mars_csv.parent), "-o", str(out), "--pattern", "*.csv"]) == 0

    import pandas as pd

    frame = pd.read_csv(out)
    # The challenge requires filename plus a time column to be scoreable.
    assert {"filename", "time_rel(sec)", "mq_type"} <= set(frame.columns)
    assert len(frame) > 0


def test_cli_reports_a_missing_model_cleanly(tmp_path, mars_csv, capsys):
    code = main(["predict", str(mars_csv), "--model", str(tmp_path / "absent.joblib")])
    assert code == 1
    assert "cosmoquake train" in capsys.readouterr().err


def test_catalog_prefers_csv_over_mseed_for_one_event(mars_csv):
    """Both exports of an event must not be catalogued twice."""
    records = list(iter_records(mars_csv.parent, "*"))
    stems = [path.with_suffix("").name for path in records]
    assert len(stems) == len(set(stems))


# --------------------------------------------------------------------------
# HTTP service
# --------------------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_model_state(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Cosmoquake" in response.text


def test_model_endpoint_describes_the_classifier(client):
    payload = client.get("/api/model").json()
    assert set(payload["classes"]) == {"impact_mq", "deep_mq", "shallow_mq"}
    assert len(payload["feature_importances"]) == 5


def test_predict_endpoint_classifies_a_record(client, mars_csv):
    with mars_csv.open("rb") as handle:
        response = client.post("/api/predict", files={"file": (mars_csv.name, handle, "text/csv")})
    assert response.status_code == 200

    payload = response.json()
    assert payload["label"]
    assert payload["record"]["compression_ratio"] > 1


def test_predict_rejects_an_unsupported_type(client):
    response = client.post("/api/predict", files={"file": ("x.txt", b"data", "text/plain")})
    assert response.status_code == 415


def test_predict_rejects_an_empty_file(client):
    response = client.post("/api/predict", files={"file": ("x.csv", b"", "text/csv")})
    assert response.status_code == 400


def test_predict_rejects_an_unparseable_csv(client):
    response = client.post(
        "/api/predict", files={"file": ("x.csv", b"alpha,beta\n1,2\n", "text/csv")}
    )
    assert response.status_code == 422


def test_upload_filename_cannot_escape_the_workspace(client, mars_csv):
    """A traversal attempt in the filename is reduced to its basename."""
    with mars_csv.open("rb") as handle:
        response = client.post(
            "/api/predict",
            files={"file": ("../../../../tmp/evil.csv", handle, "text/csv")},
        )
    assert response.status_code == 200
    assert "/" not in response.json()["source"]
