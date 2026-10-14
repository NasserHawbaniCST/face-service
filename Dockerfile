# Face service image: python slim, non-root, models baked in, healthcheck.
# Build:  docker build -t face-service .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# opencv-python-headless ships its own libs; libglib2.0-0 / libgomp1 are cheap insurance
# for the slim base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin face

WORKDIR /srv/face_service

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
# the two ONNX models (OpenCV Zoo; sha256 verified again at start-up)
COPY models ./models

USER face
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3).status == 200 else 1)"

# ONE process (replay cache + rate limits are in memory); parallelism = FACE_WORKERS engines/threads.
CMD ["uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8765", "--no-access-log"]
