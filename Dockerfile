# Usar una imagen base ligera de Python 3.12
FROM python:3.12-slim

# Evitar que Python escriba archivos .pyc y forzar salida en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PADDLE_DISABLE_SIGNAL_HANDLER=1
ENV GLOG_minloglevel=3

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y compilar librerías
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requisitos
COPY requirements.txt .

# Actualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Instalar PyTorch para CPU explícitamente primero para ahorrar espacio y evitar versiones CUDA
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de dependencias (se ignorará torch/torchvision porque ya están instalados)
RUN pip install --no-cache-dir -r requirements.txt

# Descargar de antemano el modelo de YOLO para evitar que lo descargue en cada inicio
# (si yolov8n.pt ya existe localmente se copiará en el siguiente paso)

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para iniciar la aplicación de Streamlit
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
