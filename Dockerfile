# Multi-stage production Dockerfile for lore-mcp (Task 16.1, 16.2, 18.2)

# Global uv binary stage
FROM ghcr.io/astral-sh/uv:latest AS uv-bin


# Stage 1: Dependency builder for production server (lightweight, Task 16.1)
FROM python:3.12-slim-bookworm AS builder-server

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install uv binary from official image
COPY --from=uv-bin /uv /uvx /bin/

# Install server production dependencies first for layer caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra server

# Copy application source code and series configurations
COPY src/ ./src/
COPY series/ ./series/
COPY README.md ./

ARG APP_VERSION=0.1.0-dev
ARG INGEST_VERSION=1
ENV APP_VERSION=${APP_VERSION} \
    INGEST_VERSION=${INGEST_VERSION}

# Re-sync to install the lore-mcp package itself with server extra
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server


# Stage 2: Ingestion builder (Task 16.2 - includes ingest extra with FastEmbed & scrapers)
FROM builder-server AS builder-ingest

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server --extra ingest


# Stage 3: Ingestion runtime image (Task 16.2)
FROM python:3.12-slim-bookworm AS ingest

RUN groupadd -g 10001 lore && \
    useradd -u 10001 -g lore -m -s /bin/bash lore

WORKDIR /app

COPY --from=builder-ingest --chown=lore:lore /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER lore

ENTRYPOINT ["lore"]


# Stage 4: Minimal non-root production server runtime (Default Target, Task 16.1, 18.2)
FROM python:3.12-slim-bookworm AS runtime

# Security: Create non-root unprivileged user and group
RUN groupadd -g 10001 lore && \
    useradd -u 10001 -g lore -m -s /bin/bash lore

WORKDIR /app

# Copy virtualenv and application with ownership from builder-server
COPY --from=builder-server --chown=lore:lore /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HOST=0.0.0.0

USER lore

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import os, urllib.request; port=os.environ.get('PORT', '8080'); urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz')" || exit 1

ENTRYPOINT ["lore", "serve", "--host", "0.0.0.0"]
