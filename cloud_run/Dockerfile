# Build from the repository root:
#   docker build -f cloud_run/Dockerfile -t esmi-compress .
#   docker run --rm -p 8080:8080 esmi-compress

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OpenJPEG enables real JPEG2000 via Pillow when available.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    libgomp1 \
    libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

COPY cloud_run/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY compression /app/compression
COPY svd_compression.py image_io.py ndvi.py /app/
COPY cloud_run/main.py /app/main.py

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
