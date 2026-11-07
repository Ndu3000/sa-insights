# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# SA Insights — multi-stage image
#
# Two concerns drive the shape of this file:
#
# 1. Dependency installation must be cached independently of source changes.
#    The `deps` stage copies only pyproject.toml and uv.lock, so editing Python
#    code never triggers a re-resolve.
#
# 2. The foundation model is optional. torch is ~200 MB on top of a ~300 MB
#    base, so it lives behind a build arg. Build with FM=1 for the full model
#    zoo, or FM=0 for a lean image where Chronos gracefully stands down.
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.11


# ---------------------------------------------------------------------------
# Base: shared environment and the uv binary
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Pinned literal, not an ARG: `COPY --from` cannot interpolate build args, so
# an ARG here fails with "invalid reference format".
COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uvx /usr/local/bin/uvx

WORKDIR /app


# ---------------------------------------------------------------------------
# Dependencies: cached unless the lockfile changes
# ---------------------------------------------------------------------------
FROM base AS deps

ARG WITH_FM=1

COPY pyproject.toml uv.lock ./

# No --extra-index-url here on purpose. The CPU index is declared in
# pyproject.toml ([tool.uv.index], explicit) and pinned in uv.lock, so
# resolution happens once and the image cannot disagree with the lockfile.
# Passing --extra-index-url instead is what silently re-admitted the ~2.5 GB
# CUDA stack: it is additive, so PyPI still won for `torch`.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_FM" = "1" ]; then \
        uv sync --frozen --no-install-project --no-dev --extra fm ; \
    else \
        uv sync --frozen --no-install-project --no-dev ; \
    fi

# Fail the build, not the dashboard.
#
# Every failure mode this project hit — the flag defaulting to 0, the extra
# index losing to PyPI, the model simply absent — produced an image that built
# cleanly and only surfaced as "Foundation model unavailable" in the UI days
# later. A one-line import converts that whole class into a failed build.
RUN if [ "$WITH_FM" = "1" ]; then \
        /app/.venv/bin/python -c \
          "import torch, chronos; print('torch', torch.__version__, '| chronos OK')" ; \
    fi


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM base AS runtime

ARG WITH_FM=1
LABEL org.opencontainers.image.title="sa-insights" \
      org.opencontainers.image.description="South African macroeconomic dashboard with forecasting and model tracking" \
      org.opencontainers.image.licenses="MIT" \
      sa-insights.foundation-model="${WITH_FM}"

# Bring the pre-built virtualenv across; nothing else from `deps` is needed.
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# Application code last: this is the layer that actually changes between builds.
COPY sa_insights/ ./sa_insights/
COPY scripts/ ./scripts/
COPY semantic/ossie-schema.json ./semantic/ossie-schema.json
COPY app.py ./
COPY .streamlit/ ./.streamlit/

# Writable data directory for DuckDB. Declared as a volume so the database
# survives container replacement.
#
# Note the ownership strategy: create the directories and hand them to the app
# user *before* copying the virtualenv. A recursive chown over /app after the
# copy rewrites ownership for tens of thousands of files and added nearly five
# minutes to the build on a cold cache.
ENV DATA_DIR=/data
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data /home/app/.cache \
 && chown -R app:app /data /home/app
VOLUME ["/data"]

USER app
EXPOSE 8501

# Streamlit ships a health endpoint; use it rather than probing the root page.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
