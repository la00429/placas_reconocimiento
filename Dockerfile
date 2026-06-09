# ──────────────────────────────────────────────
# Imagen base: Python 3.12 slim
# ──────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# ──────────────────────────────────────────────
# Dependencias del sistema para OpenCV headless
# ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────
# Dependencias Python
# PyTorch CPU primero — evita descargar la versión
# CUDA de 2+ GB que pip elegiría por defecto
# ──────────────────────────────────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────
# Pre-descargar modelos de EasyOCR en build time
# Así el contenedor arranca rápido en Railway
# ──────────────────────────────────────────────
RUN python -c "\
import easyocr; \
easyocr.Reader(['en'], gpu=False); \
print('Modelos EasyOCR descargados')" \
    || echo "Descarga pospuesta a runtime"

# ──────────────────────────────────────────────
# Código — va DESPUÉS de dependencias para
# aprovechar caché Docker en redeploys
# ──────────────────────────────────────────────
COPY . .

EXPOSE 8501

# Railway lo usa para saber cuándo el contenedor está listo
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app_streamlit.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=10"]
