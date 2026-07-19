# Sainik Sahayak rag-service — Mumbai deployment (Fly.io blr / ap-south-1 VM).
FROM python:3.11-slim

# System deps: Tesseract with Hindi + English traineddata for the OCR fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "FlagEmbedding>=1.3"

# Pre-download BGE-M3 + reranker into the image so first request is warm and the
# container needs no model-hub egress at runtime (air-gap friendly).
ENV HF_HOME=/srv/.hf
RUN python -c "from FlagEmbedding import BGEM3FlagModel, FlagReranker; \
    BGEM3FlagModel('BAAI/bge-m3'); FlagReranker('BAAI/bge-reranker-v2-m3')" || \
    echo "model pre-download skipped (no network at build); will fetch on first use"

COPY app ./app
COPY cli.py .

ENV RAG_EMBEDDING_BACKEND=bge \
    RAG_RERANK_BACKEND=bge

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
