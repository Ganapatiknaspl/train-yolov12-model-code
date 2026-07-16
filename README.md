# YOLOv12L Aircraft Detection

A custom YOLOv12L object detection model trained for aerial object detection.

## Classes

| ID | Class |
|----|---------|
|0|Aeroplane|
|1|Drone|
|2|Helicopter|
|3|Bird|

---

# Dataset Structure

```
AerialDataset/

    data.yaml

    train/
        images/
        labels/

    valid/
        images/
        labels/

    test/
        images/
        labels/
```

---

# Requirements

- Python 3.10+
- NVIDIA GPU
- CUDA 12.x
- PyTorch

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your_username>/<repository_name>.git
```

Move into the project

```bash
cd <repository_name>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run

```bash
python model.py
```

The script will automatically:

- Load YOLOv12L
- Load the dataset
- Start training
- Save checkpoints
- Save plots
- Save the best model
- Display training duration

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
|Model|YOLOv12L|
|Image Size|1024|
|Batch Size|16|
|Epochs|100|
|Optimizer|AdamW|
|Learning Rate|0.001|
|Weight Decay|0.001|
|Mixed Precision|Enabled|
|Seed|42|

---

# Data Augmentation

- Mosaic
- MixUp
- Copy Paste
- HSV Augmentation
- Random Scale
- Translation
- Horizontal Flip
- Vertical Flip
- Shear
- Perspective

---

# Output

Training outputs are stored in

```
YOLO12_Training/
```

Including

- best.pt
- last.pt
- results.csv
- results.png
- confusion_matrix.png
- PR_curve.png
- F1_curve.png
- P_curve.png
- R_curve.png

---

# Dataset

Dataset should be placed inside

```
AerialDataset/
```

with

```
train/
valid/
test/
data.yaml
```

---

# Notes

- The project automatically records training start time and end time.
- Best model is saved automatically.
- Mixed Precision (AMP) is enabled.
- Deterministic training is enabled for reproducibility.

---

# Citation

If you use this project, please cite Ultralytics YOLO.

```
@software{ultralytics,
author = {Ultralytics},
title = {Ultralytics YOLO},
year = {2025}
}
```

---

