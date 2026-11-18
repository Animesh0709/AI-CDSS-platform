# ===============================================================
# Dockerfile — AI-CDSS Backend (FastAPI + PyTorch CPU)
# Compatible with Linux x86_64 Cloud Platforms (Render, HF, Fly)
# ===============================================================
FROM python:3.11-slim

# System deps (tesseract, poppler for OCR; libgomp for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

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

# Set PYTHONPATH so 'ml.*' and 'app.*' imports resolve cleanly
ENV PYTHONPATH="/app/backend:/app"
ENV PORT=10000

WORKDIR /app/backend
EXPOSE 10000
EXPOSE 8000

CMD ["python", "run_backend.py"]
