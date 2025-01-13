FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    requests \
    pydantic \
    redis \
    websockets \
    asyncio \
    minio \
    httpx \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-requests \
    opentelemetry-instrumentation-redis \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-instrumentation-httpx \
    opentelemetry-instrumentation-urllib3 \
    ujson

# Copy shared package first
COPY shared /shared
RUN pip install /shared

# Copy service files
COPY services/APIService/main.py ./

# Use uvicorn with websocket support enabled
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--ws", "auto"]