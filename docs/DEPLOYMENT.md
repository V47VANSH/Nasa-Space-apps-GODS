# Deployment

The service is one stateless container with the model baked in. It holds no
database, writes only to a per-request temporary directory, and answers
`GET /health` for readiness checks.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Listen port; most PaaS hosts inject this |
| `COSMOQUAKE_MODEL` | bundled | Path to a model artifact |
| `COSMOQUAKE_DATASET` | bundled | Path to the training table |
| `COSMOQUAKE_MAX_UPLOAD_MB` | `64` | Rejects larger uploads with HTTP 413 |

## Docker

```bash
docker build -t cosmoquake .
docker run -p 8000:8000 cosmoquake
curl http://localhost:8000/health
```

The build is two-stage: dependencies compile in the builder, and the runtime
image carries only the virtualenv, the model and the data. It runs as an
unprivileged user (uid 10001).

To serve a model you trained yourself:

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  -e COSMOQUAKE_MODEL=/app/models/my_model.joblib \
  cosmoquake
```

## Render

Create a **Web Service** from the repository, choose the **Docker** runtime and
leave the defaults; `$PORT` is injected automatically. Set the health check path
to `/health`.

Or commit this `render.yaml` and use Blueprints:

```yaml
services:
  - type: web
    name: cosmoquake
    runtime: docker
    healthCheckPath: /health
    envVars:
      - key: COSMOQUAKE_MAX_UPLOAD_MB
        value: "32"
```

## Railway

`railway up` from the repository root. The Dockerfile is detected and `$PORT`
is injected. No further configuration is required.

## Fly.io

```bash
fly launch --no-deploy      # generates fly.toml from the Dockerfile
fly deploy
```

In `fly.toml`, set the internal port to 8000 and point the HTTP check at
`/health`.

## Google Cloud Run

```bash
gcloud run deploy cosmoquake \
  --source . \
  --port 8000 \
  --allow-unauthenticated \
  --memory 1Gi
```

1 GiB is the practical floor: scikit-learn, SciPy and ObsPy together need
roughly 400 MB resident before any request arrives.

## Hugging Face Spaces

Create a Space with the **Docker** SDK, push the repository, and add
`app_port: 8000` to the Space's `README.md` front matter.

## Behind a reverse proxy

Uploads are streamed and capped in the application, but the proxy has its own
limit — raise it to match:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    client_max_body_size 64M;
    proxy_read_timeout 120s;   # a one-hour record takes a few seconds to analyse
}
```

## Scaling notes

- **Stateless.** Scale horizontally; no shared state between instances.
- **Startup.** The model loads once at startup, not per request. Allow ~10 s
  before the first health check passes (the container's own healthcheck uses a
  20 s start period).
- **Memory.** Roughly 400 MB baseline plus about 3× the record size while a
  request is being analysed. 1 GiB per instance is comfortable.
- **CPU-bound.** The FFT dominates. One worker per core; add instances rather
  than threads.
- **Missing model.** If no artifact can be found the service still starts and
  serves `/health` with `model_loaded: false`, and prediction returns HTTP 503
  with instructions. It does not crash-loop.
