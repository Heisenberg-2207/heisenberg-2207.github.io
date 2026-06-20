# Wheelchair Localisation and Mapping

> BioNeX Lab Undergraduate Research Project · IIT Madras · Jan–Jun 2024  
> Guide: Dr. Manish Anand, Dept. of Mechanical Engineering

Computer vision pipeline for wheelchair detection and automatic docking, using YOLOv8 trained on a custom-annotated dataset.

## Problem

Semi-autonomous wheelchairs need to know where they are relative to docking stations. Traditional LIDAR-based localization is expensive. This project asks: **can a single monocular camera, paired with a trained object detector, provide sufficient localization for autonomous docking?**

## What's Here

```
Localisation_and_Mapping/
├── yolo.ipynb          ← Training + inference notebook
├── UGRC_Report.pdf     ← Full undergraduate research report
├── Result.png          ← Inference result example
├── Results/
│   ├── detect_wheel.jpg         ← Detection output
│   ├── estimate_distance.jpg    ← Distance estimation result
│   └── video_proof.mp4          ← Live demo video
└── train/
    ├── weights/best.pt          ← Best YOLOv8 checkpoint
    ├── results.csv              ← Training metrics
    ├── confusion_matrix.png     ← Evaluation
    ├── F1_curve.png / PR_curve.png / ...
    └── val_batch*.jpg           ← Validation predictions
```

## Approach

### Dataset
- **500+ images** annotated manually (bounding boxes around wheelchair wheels)
- Sources: open-source wheelchair image datasets + collected images
- 3 classes: `left_wheel`, `right_wheel`, `background`
- 80/20 train/val split

### Model
**YOLOv8n** (nano variant) — small, fast, suitable for real-time inference on embedded hardware.

Training:
```
Epochs: 25
Batch size: 16
Image size: 640×640
Optimizer: auto (Ultralytics default selection)
```

### Distance Estimation
Wheel diameter is known (standard ~60 cm). Using the bounding box pixel width and camera intrinsics:

```
distance = (focal_length × real_width) / pixel_width
```

Achieved position estimates within **±2 cm** uncertainty.

### Pipeline Integration
Detection output feeds into a semi-autonomous wheelchair controller:
1. Detect wheels in frame
2. Estimate distance and lateral offset
3. Send correction commands to motor controller
4. Dock when distance < threshold

## Results

| Metric | Value |
|---|---|
| mAP@0.5 | 83.1% |
| Position uncertainty | ±2 cm |
| Inference speed | ~25 FPS (laptop GPU) |

Despite the relatively small training set, the model generalizes well across most common wheelchair types, not just the ones represented in training.

## Technologies

`Python` · `YOLOv8 (Ultralytics)` · `OpenCV` · `PyTorch` · `NumPy` · `Matplotlib`

## How to Run

```bash
pip install ultralytics opencv-python numpy
```

Open `yolo.ipynb` and run all cells. The best model weights are in `train/weights/best.pt`.

For inference on a new image:
```python
from ultralytics import YOLO
model = YOLO("train/weights/best.pt")
results = model("your_image.jpg")
results[0].show()
```
