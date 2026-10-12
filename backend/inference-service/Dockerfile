FROM python:3.12-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY segmentation-core /build/segmentation-core
COPY inference-service/requirements-api.txt /build/requirements.txt
RUN pip install /build/segmentation-core -r /build/requirements.txt && rm -rf /build

RUN useradd --create-home --uid 1000 segmentation \
    && mkdir -p /storage \
    && chown -R segmentation:segmentation /storage

COPY inference-service/inference_service ./inference_service

USER segmentation
EXPOSE 8000

CMD ["uvicorn", "inference_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*", "--timeout-graceful-shutdown", "30"]


FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS worker

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY segmentation-core /build/segmentation-core
COPY inference-service/requirements-worker.txt /build/requirements.txt
RUN pip install /build/segmentation-core -r /build/requirements.txt && rm -rf /build

RUN useradd --create-home --uid 1000 segmentation \
    && mkdir -p /storage /artifacts \
    && chown -R segmentation:segmentation /storage

COPY inference-service/inference_service ./inference_service

USER segmentation

CMD ["python", "-m", "inference_service.worker"]
