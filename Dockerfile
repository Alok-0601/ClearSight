# Container image for the FastAPI backend, for any Docker host.
# Build: docker build -t clearsight-api .
# Run:   docker run -p 8000:8000 --env-file .env clearsight-api
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r pyproject.toml "uvicorn[standard]"

COPY preprocess.py ./
COPY Models ./Models
COPY nltk_data ./nltk_data
COPY backend ./backend

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
