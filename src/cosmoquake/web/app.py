"""FastAPI service wrapping the analysis pipeline.

Endpoints:
    GET  /            upload page
    GET  /health      liveness probe, reports whether the model loaded
    GET  /api/model   model metadata, classes and cross-validated scores
    POST /api/predict classify an uploaded .csv or .mseed record
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from cosmoquake import __version__
from cosmoquake.model import CosmoquakeModel, ModelError, load_model
from cosmoquake.pipeline import analyze
from cosmoquake.waveform import CSV_SUFFIXES, MSEED_SUFFIXES, WaveformError

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

#: Refuse anything larger, so a stray upload cannot exhaust container memory.
#: One hour of 20 Hz InSight data is about 4 MB as CSV.
MAX_UPLOAD_BYTES = int(os.environ.get("COSMOQUAKE_MAX_UPLOAD_MB", "64")) * 1024 * 1024

#: Read the body in chunks rather than all at once, so the size limit can be
#: enforced before a large file is fully in memory.
CHUNK_BYTES = 1024 * 1024

ALLOWED_SUFFIXES = frozenset(CSV_SUFFIXES + MSEED_SUFFIXES)

#: Populated at startup; ``None`` when no artefact could be loaded, which
#: keeps the service up to serve /health and a clear error instead of
#: crash-looping in a deployment.
_model: CosmoquakeModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than per request."""
    global _model
    try:
        # DEFAULT_MODEL_PATH already honours the COSMOQUAKE_MODEL override.
        _model = load_model()
        logger.info("loaded model with classes %s", ", ".join(_model.classes))
    except ModelError as error:
        _model = None
        logger.error("could not load model: %s", error)
    yield
    _model = None


app = FastAPI(
    title="Cosmoquake Analyzer",
    description=(
        "Detect and classify seismic events in Apollo and InSight records. "
        "Upload a .csv or .mseed trace to get an arrival time, five "
        "frequency-domain features and a predicted event class."
    ),
    version=__version__,
    lifespan=lifespan,
)


def _require_model() -> CosmoquakeModel:
    """Return the loaded model or fail with a 503."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model is loaded. Train one with `cosmoquake train`, or set "
                "COSMOQUAKE_MODEL to the path of an artefact."
            ),
        )
    return _model


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the upload page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    """Report service and model status for container health checks."""
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": _model is not None,
    }


@app.get("/api/model")
async def model_info() -> dict[str, Any]:
    """Describe the loaded model, its classes and its measured performance."""
    model = _require_model()
    return {
        "classes": list(model.classes),
        "features": list(model.feature_names),
        "feature_importances": model.feature_importances(),
        "training_range": {
            name: {"min": low, "max": high}
            for name, (low, high) in model.training_range().items()
        },
        "metadata": model.metadata,
    }


async def _save_upload(upload: UploadFile, directory: Path) -> Path:
    """Stream an upload to disk, enforcing the size limit as it goes."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix or '(none)'}; expected "
            f"one of {sorted(ALLOWED_SUFFIXES)}",
        )

    # Keep the caller's basename only: a filename like "../../etc/passwd"
    # must not escape the temporary directory.
    destination = directory / Path(upload.filename or "upload").name
    written = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(CHUNK_BYTES):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )
            handle.write(chunk)

    if written == 0:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    return destination


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(..., description="A .csv or .mseed seismic record"),
) -> JSONResponse:
    """Classify one uploaded seismic record."""
    model = _require_model()

    with tempfile.TemporaryDirectory(prefix="cosmoquake-") as workspace:
        path = await _save_upload(file, Path(workspace))
        try:
            result = analyze(path, model=model)
        except WaveformError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:  # pragma: no cover - unexpected parse failures
            logger.exception("analysis failed for %s", file.filename)
            raise HTTPException(
                status_code=422, detail=f"could not analyse this record: {error}"
            ) from error

    return JSONResponse(result.to_dict())
