# syntax=docker/dockerfile:1.4
FROM forte-cash-base:1.0

WORKDIR /app
COPY api.py engine.py violation_rules_verified.csv yolov8n.pt ./

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -f http://localhost:8000/ || exit 1
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
