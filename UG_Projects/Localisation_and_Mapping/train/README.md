# Training Run

Ultralytics YOLOv8n training output for the wheelchair-wheel detector (25 epochs, batch size 16, image size 640×640 — see `args.yaml` for the full config).

| File | Description |
|---|---|
| `weights/best.pt`, `weights/last.pt` | Best (highest mAP) and final-epoch model checkpoints |
| `results.csv` / `results.png` | Per-epoch loss and metric curves (box/cls/dfl loss, precision, recall, mAP50, mAP50-95) |
| `confusion_matrix.png` / `confusion_matrix_normalized.png` | 3-class confusion matrix (left wheel / right wheel / background) |
| `F1_curve.png`, `PR_curve.png`, `P_curve.png`, `R_curve.png` | F1, precision-recall, precision, and recall vs. confidence-threshold curves |
| `labels.jpg` / `labels_correlogram.jpg` | Dataset label distribution and correlation plots |
| `train_batch*.jpg` | Sample training batches with augmentations and ground-truth boxes |
| `val_batch*_labels.jpg` / `val_batch*_pred.jpg` | Validation batches: ground truth vs. model predictions |
| `args.yaml` | Full Ultralytics training configuration |

Best result: **mAP@0.5 = 83.1%** at epoch 25 (final epoch — see `results.csv`).
