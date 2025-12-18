# Milestone 1: Video Analytics Pipeline
# Basic Detection + Tracking with YOLOv8 and OC-SORT

FROM python:3.10-slim

LABEL maintainer="Bhanu Prakash Vangala, Nolan Rink"
LABEL description="Milestone 1 - Video Analytics Pipeline"

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
COPY milestone1.ipynb .

# Copy model weights (if not downloading)
# COPY yolov8n.pt .

# Download YOLOv8n model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create directories
RUN mkdir -p /app/output /app/Dataset

# Environment variable for dataset path
ENV DATASET_PATH=/app/Dataset/MOT17/train/MOT17-04-DPM/img1

# Default command - run Jupyter notebook
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "milestone1.ipynb"]
