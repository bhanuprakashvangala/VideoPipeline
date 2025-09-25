# Video Object Detection and Tracking Pipeline
# Author: Bhanuprakash Vangala
# Docker Hub: vangalabhanuprakash/video-pipeline

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    curl \
    wget \
    git \
    unzip \
    nano \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file
COPY requirements-video.txt .

# Install Python dependencies with NumPy 2.x compatibility
RUN pip install --upgrade pip && \
    pip install numpy>=2.0.0 && \
    pip install scipy filterpy && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install transformers ultralytics opencv-python Pillow pandas matplotlib && \
    pip install jupyterlab notebook ipywidgets

# Copy project files
COPY README.md .
COPY video_pipeline_complete.ipynb .

# Copy MOT17 dataset (only the single sequence we're using)
COPY MOT17/train/MOT17-04-DPM/ /app/MOT17/train/MOT17-04-DPM/

# Download YOLOv8 model weights
RUN wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O /app/yolov8n.pt

# Create results directory
RUN mkdir -p /app/results

# Expose Jupyter port
EXPOSE 8888

# Set environment variables for Jupyter
ENV JUPYTER_ENABLE_LAB=yes
ENV PYTHONUNBUFFERED=1

# Default command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]