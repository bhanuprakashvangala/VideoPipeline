# Combined - Video Analytics Pipeline

## All Milestones in One Package

This folder contains all milestones combined with a single Docker setup.

### Contents

| File | Description |
|------|-------------|
| `milestone1.ipynb` | Basic Detection + Tracking Pipeline |
| `milestone2.ipynb` | Pipeline Optimization with Model Variants |
| `milestone3_complete.ipynb` | INFaaS Optimizer + Agentic + Assertions |
| `t4.py` | T4 Agentic Framework with LLM Integration |
| `Dockerfile` | Combined Docker image |
| `requirements.txt` | All Python dependencies |

---

## Quick Start

### 1. Build Docker Image
```bash
docker build -t videopipeline .
```

### 2. Run with Dataset
```bash
# Start Jupyter server
docker run -p 8888:8888 -v /path/to/Dataset:/app/Dataset videopipeline

# Access at http://localhost:8888
```

### 3. Run Specific Milestone
```bash
# Milestone 1
docker run -v ./Dataset:/app/Dataset videopipeline \
    jupyter nbconvert --execute --inplace milestone1.ipynb

# Milestone 2
docker run -v ./Dataset:/app/Dataset videopipeline \
    jupyter nbconvert --execute --inplace milestone2.ipynb

# Milestone 3
docker run -v ./Dataset:/app/Dataset videopipeline \
    jupyter nbconvert --execute --inplace milestone3_complete.ipynb
```

### 4. Run T4 Video Demo
```bash
docker run -v ./Dataset:/app/Dataset -v ./output:/app/output videopipeline \
    python t4.py
```

---

## Milestones Overview

### Milestone 1: Basic Pipeline
- YOLOv8 Object Detection
- OC-SORT Multi-Object Tracking
- Performance Metrics Collection

### Milestone 2: Pipeline Optimization
- Multiple Model Variants (YOLOv8n, s, m, l)
- Horizontal Scaling Analysis
- Latency vs Accuracy Trade-offs
- SLA Compliance Testing

### Milestone 3: Automated Optimizer
- **T1**: Pipeline Accuracy Definition
- **T2**: INFaaS Optimization Problem
- **T3**: ILP + Rule-based Optimizer
- **T4**: Agentic Framework with LLM
- **Assertions**: 6 Pipeline Validation Checks

---

## Requirements

- Docker 20.10+
- NVIDIA GPU (optional, for faster inference)
- 8GB+ RAM
- MOT17 Dataset
