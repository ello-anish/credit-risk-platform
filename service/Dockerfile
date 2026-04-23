# FastAPI scoring service image.
# Build from repo root:   docker build -f service/Dockerfile -t credit-risk-service .
FROM python:3.11-slim

WORKDIR /app

# System deps (psycopg2 needs libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the service needs — the model artefacts live in a mounted
# volume in production (and are baked in at dev time for docker compose).
COPY pipeline /app/pipeline
COPY models /app/models
COPY reconciliation /app/reconciliation
COPY stress /app/stress
COPY ecl /app/ecl
COPY service /app/service
COPY config.yml /app/config.yml

ENV PYTHONPATH=/app
ENV DB_HOST=postgres

EXPOSE 8000
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
