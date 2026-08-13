# Cosmoquake Analyzer web service.
#
#   docker build -t cosmoquake .
#   docker run -p 8000:8000 cosmoquake
#
# Built in two stages so the runtime image carries no compilers: obspy and
# scipy pull in sizeable build toolchains that are useless once wheels exist.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COSMOQUAKE_MODEL=/app/models/cosmoquake_rf.joblib \
    COSMOQUAKE_DATASET=/app/data/moonquakes.csv

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY models ./models
COPY data ./data

# Run unprivileged: the service only ever reads its model and writes to a
# temporary directory it creates per request.
RUN useradd --create-home --uid 10001 cosmoquake && chown -R cosmoquake:cosmoquake /app
USER cosmoquake

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# $PORT is respected so the same image runs on Render, Railway and Fly.io.
CMD ["sh", "-c", "uvicorn cosmoquake.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
