# syntax=docker/dockerfile:1.4
ARG BASE_IMAGE=forte-cash-base:1.0
FROM ${BASE_IMAGE}

WORKDIR /app
COPY api.py engine.py production.py violation_rules_verified.csv yolov8n.onnx ./

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live', timeout=3)" || exit 1
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
