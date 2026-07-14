# Document Layout Detection with YOLOv8

A supervised object-detection pipeline that locates structural elements — titles, section markers, page numbers, and other layout regions — in scanned/archival documents using **YOLOv8**. The pipeline covers manual annotation, augmentation, training, hyperparameter optimization with Optuna, and evaluation.

> _Add course / institution / author details here._

---

## Overview

The goal is to detect and classify layout regions in document images with a supervised, end-to-end detector. YOLOv8 is well-suited to this: it learns directly from labeled data, supports multiple object classes and complex layouts, and runs fast enough for real-time, large-scale processing without heavy manual feature engineering.

The main trade-offs are the usual supervised ones — it needs a substantial amount of labeled data, it's sensitive to annotation quality (mislabeled boxes degrade performance), it can overfit without proper augmentation/regularization, and training/tuning is computationally expensive.

**Classes detected** — e.g. `Title`, `Section_Marker`, `Page_Number`, … (`nc = 6` classes total; edit to match your dataset).

## Pipeline

```
Raw document images
        │  LabelImg (manual bounding-box annotation)
        ▼
Albumentations augmentation ──► Custom Dataset + PyTorch DataLoader
        │                              │
        │  dataset.yaml                ▼
        └──────────────────►  YOLOv8m training  ──►  Optuna HPO
                                       │
                                       ▼
                          Evaluation (P / R / mAP)
```

## Data Labeling

Bounding boxes are annotated manually with **LabelImg**, producing one `.txt` label file per image in YOLO format. Label quality is critical — inconsistent annotations introduce label noise that directly limits achievable accuracy.

## Preprocessing & Augmentation

Applied via **Albumentations** during data access:

**1. Spatial transformations**
- Resize all images to **640×640** (YOLOv8's default input size)
- Horizontal flip, applied randomly (~50% probability) for left–right variability
- Shift / scale / rotate with moderate limits to simulate real-world misalignment

**2. Photometric adjustments**
- Random brightness/contrast (~30% probability) to simulate lighting variation
- Gaussian noise to mimic low-resolution scans or degraded document quality

**3. Normalization & conversion**
- Normalize pixel values using ImageNet mean and standard deviation
- Convert to PyTorch tensors (`ToTensor`) for training

## System Architecture

**1. Custom Dataset class**
- Automatically pairs each image with its `.txt` label file
- Applies Albumentations transforms on access
- Keeps labels consistent after spatial/photometric changes

**2. PyTorch DataLoader**
- Training: batch size 16, shuffling enabled (improves generalization)
- Validation: batch size 16, shuffling disabled (deterministic, comparable evaluation)

**3. YAML configuration (`dataset.yaml`)**
- `train` — path to training image folder
- `val` — path to validation image folder
- `nc` — number of object classes (e.g. 6)
- `names` — list of class labels (e.g. `"Title"`, `"Section_Marker"`, `"Page_Number"`, …)

## Training

- **Model:** YOLOv8m (medium architecture), pretrained weights from the Ultralytics repository
- **Mixed precision:** `half=True` to reduce memory use and speed up training
- **Epochs:** ~50 over the full dataset
- **Validation:** run after each epoch to monitor progress
- **Checkpointing:** best model state saved by validation performance

### Evaluation metrics

| Metric | Meaning |
|---|---|
| **Precision** | Proportion of true positives among predicted positives |
| **Recall** | Proportion of detected objects among all ground-truth objects |
| **mAP@50** | Mean average precision at IoU threshold 0.50 |
| **mAP@50–95** | Mean average precision averaged across IoU thresholds (0.50→0.95, step 0.05) |

## Hyperparameter Optimization (Optuna)

Optuna searches over the following domains:

| Hyperparameter | Search domain |
|---|---|
| Learning rate (`lr0`) | 1e-5 → 1e-1 |
| Momentum | 0.85 → 0.97 |
| Weight decay | 1e-6 → 1e-2 |
| Batch size | {8, 16, 32} |

## Results

| Metric | Score |
|---|---|
| **Precision** | 87% |
| **Recall** | 66% |
| **mAP@50** | 67% |
| **mAP@50–95** | 52% |

Precision of 87% reflects accurate positive detections; recall of 66% reflects the model's ability to find all relevant regions on a page. The gap between mAP@50 (67%) and mAP@50–95 (52%) indicates boxes are found reliably at a loose IoU threshold but localize less tightly at stricter ones.

> Model speed / inference timings are reported in the accompanying slides — add the concrete numbers here if you want them in the README.

## Limitations & Future Directions

**Limitations**
- **Intra-dataset variance** — wide variation in document layout, resolution, and typography makes robust generalization across diverse archival materials hard.
- **Annotation inconsistencies** — label noise from imperfect manual annotation caps accuracy and reliability.

**Future directions**
- **Active learning** for dynamic annotation refinement — iteratively improving the training set by prioritizing the most informative samples for labeling.

## Setup

### Prerequisites
- Python 3.9+
- PyTorch
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Albumentations, Optuna, LabelImg

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Usage

```bash
# 1. Annotate images with LabelImg (YOLO format)
# 2. Configure dataset.yaml (paths, nc, names)

# Train
yolo detect train model=yolov8m.pt data=dataset.yaml epochs=50 imgsz=640 half=True

# Validate
yolo detect val model=runs/detect/train/weights/best.pt data=dataset.yaml

# Predict
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images
```

> Commands above are illustrative — adjust to match your actual scripts and repo layout.

## Authors

> _Add author names and affiliation here._
