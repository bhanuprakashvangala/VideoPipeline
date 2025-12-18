# Milestone 3: Automated Optimizer with Assertions
# INFaaS Optimization + Agentic Framework + Pipeline Assertions

FROM python:3.10-slim

LABEL maintainer="Bhanu Prakash Vangala, Nolan Rink"
LABEL description="Milestone 3 - INFaaS Optimizer + Agentic Framework + Assertions"

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
COPY milestone3_complete.ipynb .
COPY t4.py .

# Download YOLOv8 model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create directories
RUN mkdir -p /app/output /app/Dataset

# Environment variable for dataset path
ENV DATASET_PATH=/app/Dataset/MOT17/train/MOT17-04-DPM/img1

# Environment variables for LLM API (optional)
ENV LLM_API_URL=""
ENV LLM_API_KEY=""

# Default command
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "milestone3_complete.ipynb"]
