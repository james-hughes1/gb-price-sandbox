FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# LightGBM needs OpenMP at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    GBPS_DATA_DIR=/app/data

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY app ./app
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501
# Serve the app. Set PRETRAIN_ON_START=1 to (re)generate the Battery page's forecasts first
# (useful locally; on Cloud Run a scheduled job does it instead, see deploy/).
ENV PRETRAIN_ON_START=0
CMD ["sh", "-c", "if [ \"$PRETRAIN_ON_START\" = 1 ]; then python -m gb_price_sandbox.pretrain; fi; exec streamlit run app/Home.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true"]
