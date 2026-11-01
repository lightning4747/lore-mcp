# Multi-stage production Dockerfile for lore-mcp (Task 18.2)

# Stage 1: Dependency builder using astral-sh/uv
FROM ghcr.io/astral-sh/uv:latest AS uv-bin
FROM python:3.12-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install uv binary from official image
COPY --from=uv-bin /uv /uvx /bin/

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra server --extra ingest

# Copy source files, configurations, and assets
COPY src/ ./src/
COPY series/ ./series/
COPY README.md ./

ARG APP_VERSION=0.1.0-dev
ARG INGEST_VERSION=1
ENV APP_VERSION=${APP_VERSION} \
    INGEST_VERSION=${INGEST_VERSION}

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server --extra ingest


# Stage 2: Minimal non-root production runtime
FROM python:3.12-slim-bookworm AS runtime

# Security: Create non-root unprivileged user and group
RUN groupadd -g 10001 lore && \
    useradd -u 10001 -g lore -m -s /bin/bash lore

WORKDIR /app

# Copy virtualenv and application with ownership
COPY --from=builder --chown=lore:lore /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HOST=0.0.0.0

USER lore

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')" || exit 1

ENTRYPOINT ["lore", "serve", "--host", "0.0.0.0", "--port", "8080"]
