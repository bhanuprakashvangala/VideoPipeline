# Milestone 2: Pipeline Optimization
# Model Variants, Scaling, and Performance Analysis

FROM python:3.10-slim

LABEL maintainer="Bhanu Prakash Vangala, Nolan Rink"
LABEL description="Milestone 2 - Pipeline Optimization with Model Variants"

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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY milestone2.ipynb .

# Download all YOLOv8 model variants
RUN python -c "from ultralytics import YOLO; \
    YOLO('yolov8n.pt'); \
    YOLO('yolov8s.pt'); \
    YOLO('yolov8m.pt'); \
    YOLO('yolov8l.pt')"

# Create directories
RUN mkdir -p /app/output /app/Dataset

# Environment variable for dataset path
ENV DATASET_PATH=/app/Dataset/MOT17/train/MOT17-04-DPM/img1

# Environment variables for scaling
ENV BATCH_SIZE=1
ENV NUM_REPLICAS=1
ENV MODEL_VARIANT=yolov8n

# Default command
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "milestone2.ipynb"]
