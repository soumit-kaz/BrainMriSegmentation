FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY segmentation-core /build/segmentation-core
COPY training-service/requirements.txt /build/requirements.txt
RUN pip install /build/segmentation-core -r /build/requirements.txt && rm -rf /build

RUN useradd --create-home --uid 1000 segmentation \
    && mkdir -p /data /artifacts /cache \
    && chown -R segmentation:segmentation /artifacts /cache

COPY training-service/training_service ./training_service

USER segmentation
EXPOSE 8001

CMD ["uvicorn", "training_service.main:app", "--host", "0.0.0.0", "--port", "8001", "--timeout-graceful-shutdown", "30"]
