# ═══════════════════════════════════════════════════════════════
# Dockerfile — AI-CDSS Backend (FastAPI + PyTorch)
# Optimized for Hugging Face Spaces (Free 16 GB RAM) and Cloud Deployments
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim

# System deps (tesseract, poppler for OCR; libgomp for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY backend/ ./backend/
COPY ml/models/ ./ml/models/
COPY ml/ecg_classification/ ./ml/ecg_classification/
COPY ml/scripts/ ./ml/scripts/
COPY rag/ ./rag/

# Ensure uploads directory exists
RUN mkdir -p ./backend/uploads

# Set PYTHONPATH so 'ml.*' and 'app.*' imports resolve anywhere
ENV PYTHONPATH="/app/backend:/app"
ENV PORT=7860

WORKDIR /app/backend
EXPOSE 7860
EXPOSE 8000

CMD ["python", "run_backend.py"]
