# Tables for Manuscript

## Table 1. Detection Dataset Summary

| Split | Images | Object Annotations |
| --- | ---: | ---: |
| Train | 224 | 4,757 |
| Validation | 52 | 721 |
| Test | 45 | 1,134 |
| Total | 321 | 6,612 |

## Table 2. Main Detection and Counting Results

| Method | Type | Test MAE | Test RMSE | Test R2 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Extra Trees Count Regression | Classical | 7.03 | 16.10 | 0.823 | - | - | - | - |
| HistGradientBoosting Count Regression | Classical | 7.35 | 14.37 | 0.859 | - | - | - | - |
| YOLOv8n (20 epochs, GPU) | Deep learning | 8.58 | 18.26 | 0.772 | 0.780 | 0.876 | 0.864 | 0.702 |
| YOLO11n (20 epochs, GPU) | Deep learning | 8.60 | 19.81 | 0.732 | 0.769 | 0.856 | 0.851 | 0.685 |
| YOLO11s (20 epochs, GPU) | Deep learning | 10.80 | 26.75 | 0.512 | 0.783 | 0.862 | 0.850 | 0.712 |
| Watershed Counting | Classical | 14.80 | 32.77 | 0.267 | - | - | - | - |

## Table 3. Key Narrative Points

| Finding | Evidence |
| --- | --- |
| Best classical counting MAE | Extra Trees achieved the lowest MAE among the classical counting methods. |
| Best classical RMSE and R2 | HistGradientBoosting achieved the lowest RMSE and the highest classical R2. |
| Best end-to-end detector-counter | YOLOv8n produced the best joint balance between detection and counting. |
| YOLO11 trade-off | YOLO11s slightly improved localization quality but degraded counting accuracy compared with YOLO11n and YOLOv8n. |
