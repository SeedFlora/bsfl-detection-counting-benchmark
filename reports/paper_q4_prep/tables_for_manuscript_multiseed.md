# Tables for Manuscript (Strengthened Version)

## Table 1. Detection Dataset Summary

| Split | Images | Object Annotations |
| --- | ---: | ---: |
| Train | 224 | 4,757 |
| Validation | 52 | 721 |
| Test | 45 | 1,134 |
| Total | 321 | 6,612 |

## Table 2. Multi-seed Detector Comparison

| Model | Seeds | Test Count MAE (mean +- std) | Test Count R2 (mean +- std) | Test mAP50 (mean +- std) | Test mAP50-95 (mean +- std) |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 3 | 7.99 +- 1.71 | 0.733 +- 0.128 | 0.848 +- 0.003 | 0.682 +- 0.002 |
| YOLOv8n | 3 | 10.08 +- 1.55 | 0.651 +- 0.129 | 0.860 +- 0.010 | 0.697 +- 0.010 |

## Table 3. Classical and Best Deep Results

| Method | Type | Test MAE | Test RMSE | Test R2 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Extra Trees Count Regression | Classical | 7.03 | 16.10 | 0.823 | - | - | - | - |
| HistGradientBoosting Count Regression | Classical | 7.35 | 14.37 | 0.859 | - | - | - | - |
| YOLO11n Seed 2 | Deep learning | 6.07 | 14.27 | 0.861 | 0.753 | 0.876 | 0.845 | 0.682 |
| YOLOv8n Best Single Run | Deep learning | 8.58 | 18.26 | 0.772 | 0.780 | 0.876 | 0.864 | 0.702 |

## Table 4. Density-wise Counting Error

| Density Bin | YOLO11n MAE (mean) | YOLOv8n MAE (mean) |
| --- | ---: | ---: |
| Low | 2.38 | 2.60 |
| Medium | 5.04 | 6.38 |
| High | 16.56 | 21.27 |
