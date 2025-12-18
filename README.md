# Video Analytics Pipeline

## Multi-Object Detection and Tracking with Automated Optimization

**Authors:** Bhanu Prakash Vangala, Nolan Rink
**Course:** CS 8001 - Radiant Lab, University of Missouri

---

## Project Structure

```
VideoPipeline/
├── Dataset/                    # MOT17 Dataset
│   └── MOT17/
│       └── train/
│           └── MOT17-04-DPM/
│
├── Milestone1/                 # Basic Pipeline
│   ├── milestone1.ipynb        # Detection + Tracking
│   ├── Dockerfile
│   └── requirements.txt
│
├── Milestone2/                 # Pipeline Optimization
│   ├── milestone2.ipynb        # Model Variants + Scaling
│   ├── Dockerfile
│   └── requirements.txt
│
├── milestone3/                 # Automated Optimizer
│   ├── milestone3_complete.ipynb  # INFaaS + Assertions
│   ├── t4.py                   # Agentic Framework + LLM
│   ├── Dockerfile
│   └── requirements.txt
│
└── Combined/                   # All Milestones Combined
    ├── milestone1.ipynb
    ├── milestone2.ipynb
    ├── milestone3_complete.ipynb
    ├── t4.py
    ├── Dockerfile              # Single Docker for all
    └── requirements.txt        # Combined dependencies
```

---

## Milestones

### Milestone 1: Video Analytics Pipeline
- **Object Detection**: YOLOv8 (nano to large variants)
- **Object Tracking**: OC-SORT algorithm
- **Metrics**: FPS, latency, detection/tracking counts

### Milestone 2: Pipeline Optimization
- **Model Variants**: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l
- **Horizontal Scaling**: 1-4 replicas
- **Analysis**: Latency vs Accuracy vs Cost trade-offs
- **SLA Compliance**: 500ms latency, 20 FPS throughput

### Milestone 3: Automated Optimizer
- **T1**: Pipeline Accuracy = 0.4×Detection + 0.4×Tracking + 0.2×Consistency
- **T2**: INFaaS Optimization Problem (USENIX ATC 2021)
- **T3**: ILP Solver (PuLP) + Rule-based Optimizer
- **T4**: Agentic Framework with Real LLM (Gemma3 27B)
- **Assertions**: 6 pipeline validation checks with visualizations

---

## Quick Start

### Option 1: Run Individual Milestone
```bash
cd Milestone1
docker build -t milestone1 .
docker run -v /path/to/Dataset:/app/Dataset milestone1
```

### Option 2: Run Combined (All Milestones)
```bash
cd Combined
docker build -t videopipeline .
docker run -p 8888:8888 -v /path/to/Dataset:/app/Dataset videopipeline
# Open http://localhost:8888
```

---

## Technologies Used

| Component | Technology |
|-----------|------------|
| Detection | YOLOv8 (Ultralytics) |
| Tracking | OC-SORT |
| Optimization | PuLP ILP Solver |
| LLM | Gemma3 27B (NRP API) |
| Visualization | Matplotlib, Seaborn |
| Container | Docker |

---

## Results

### Pipeline Accuracy
- Combined Accuracy: **86.07%**
- Detection Accuracy: 71.2%
- Tracking Accuracy: 100%
- Consistency: 87.9%

### Assertion Pass Rates
| Assertion | Pass Rate |
|-----------|-----------|
| Distance (100px) | 100% |
| BBox Size (50%) | 99.8% |
| Confidence (0.3) | 100% |
| Track Continuity | 100% |
| Detection Stability | 100% |
| IoU Consistency | 100% |

---

## License

MIT License
