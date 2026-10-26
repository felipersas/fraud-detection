# syntax=docker/dockerfile:1
# Build from the repository root: docker build -f api/Dockerfile -t fraud-api .

# Stage 1: install only what the API needs (fraud-api and fraud-core, no TensorFlow).
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Dependencies first, in their own layer, so code changes do not reinstall them.
# Every member's pyproject is needed for uv to read the workspace, even unused ones.
COPY pyproject.toml uv.lock ./
COPY api/pyproject.toml api/
COPY packages/fraud-core/pyproject.toml packages/fraud-core/
COPY training/pyproject.toml training/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package fraud-api --no-install-workspace

# Then the project code, installed as regular (non-editable) packages.
COPY api/src api/src
COPY packages/fraud-core/src packages/fraud-core/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package fraud-api --no-editable

# Stage 2: a slim runtime with the virtual environment and the model bundles only.
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY models models
ENV PATH="/app/.venv/bin:$PATH" \
    FRAUD_MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["fraud-api"]
