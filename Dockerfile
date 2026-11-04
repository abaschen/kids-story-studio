# CUDA comes from the torch wheel; the NVIDIA container runtime provides the driver.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --extra gpu --no-dev

COPY storystudio ./storystudio

EXPOSE 8000
CMD ["uvicorn", "storystudio.main:app", "--host", "0.0.0.0", "--port", "8000"]
