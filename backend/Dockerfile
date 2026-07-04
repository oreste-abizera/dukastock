FROM python:3.11-slim

WORKDIR /app

# build-essential + gcc/g++ are required to compile pmdarima/prophet/xgboost
# wheels that don't ship prebuilt manylinux binaries for every platform;
# libpq-dev is required for psycopg2 (Supabase/Postgres driver).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
