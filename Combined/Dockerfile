# Combined: Video Analytics Pipeline - All Milestones
# Milestone 1: Detection + Tracking
# Milestone 2: Pipeline Optimization
# Milestone 3: INFaaS Optimizer + Agentic Framework + Assertions

FROM python:3.10-slim

LABEL maintainer="Bhanu Prakash Vangala, Nolan Rink"
LABEL description="Video Analytics Pipeline - Combined Milestones 1, 2, 3"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all notebooks and scripts
COPY milestone1.ipynb .
COPY milestone2.ipynb .
COPY milestone3_complete.ipynb .
COPY t4.py .

# Download all YOLOv8 model variants
RUN python -c "from ultralytics import YOLO; \
    print('Downloading YOLOv8 models...'); \
    YOLO('yolov8n.pt'); \
    YOLO('yolov8s.pt'); \
    YOLO('yolov8m.pt'); \
    YOLO('yolov8l.pt'); \
    print('Models downloaded successfully!')"

# Create directories
RUN mkdir -p /app/output /app/Dataset

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATASET_PATH=/app/Dataset/MOT17/train/MOT17-04-DPM/img1

# Expose Jupyter port
EXPOSE 8888

# Default command - start Jupyter notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# Alternative commands:
# Run all milestones:
# docker run -v ./Dataset:/app/Dataset videopipeline python -c "
#   import subprocess
#   subprocess.run(['jupyter', 'nbconvert', '--execute', '--inplace', 'milestone1.ipynb'])
#   subprocess.run(['jupyter', 'nbconvert', '--execute', '--inplace', 'milestone2.ipynb'])
#   subprocess.run(['jupyter', 'nbconvert', '--execute', '--inplace', 'milestone3_complete.ipynb'])
# "

# Run T4 video demo:
# docker run -v ./Dataset:/app/Dataset videopipeline python t4.py
