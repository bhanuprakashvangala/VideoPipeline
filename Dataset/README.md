# Dataset

## MOT17 - Multi-Object Tracking Benchmark

This folder contains the MOT17 dataset used for training and evaluation.

### Structure
```
Dataset/
└── MOT17/
    └── train/
        └── MOT17-04-DPM/
            └── img1/
                ├── 000001.jpg
                ├── 000002.jpg
                └── ...
```

### Download Instructions

If the dataset is not present, download from:
- Official: https://motchallenge.net/data/MOT17/
- Or use the download script:

```bash
# Download MOT17
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip -d Dataset/
```

### Usage

The dataset path in notebooks:
```python
DATASET_PATH = '../Dataset/MOT17/train/MOT17-04-DPM/img1'
```
