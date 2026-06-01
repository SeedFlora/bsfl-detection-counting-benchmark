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

## Table 3. Classical, Detector, and Reviewer-Ablation Counting Results

| Method | Type | Test MAE | Test RMSE | Test R2 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Extra Trees Count Regression | Classical | 7.03 | 16.10 | 0.823 | - | - | - | - |
| HistGradientBoosting Count Regression | Classical | 7.35 | 14.37 | 0.859 | - | - | - | - |
| YOLO11n Seed 2 | Deep learning | 6.07 | 14.27 | 0.861 | 0.753 | 0.876 | 0.845 | 0.682 |
| YOLOv8n Best Single Run | Deep learning | 8.58 | 18.26 | 0.772 | 0.780 | 0.876 | 0.864 | 0.702 |
| YOLOv8n with validation-selected confidence/NMS sweep | Reviewer ablation | 4.21 +- 0.98 | 9.62 +- 3.21 | 0.932 +- 0.046 | - | - | - | - |
| YOLO11n with minimal augmentation | Reviewer ablation | 4.27 +- 0.60 | 11.98 +- 3.34 | 0.897 +- 0.059 | 0.768 | 0.847 | 0.815 | 0.648 |

## Table 4. Density-wise Counting Error

| Density Bin | YOLO11n MAE (mean) | YOLOv8n MAE (mean) |
| --- | ---: | ---: |
| Low | 2.38 | 2.60 |
| Medium | 5.04 | 6.38 |
| High | 16.56 | 21.27 |

## Table 5. Computational and Statistical Addendum

| Item | YOLO11n | YOLOv8n | Interpretation |
| --- | ---: | ---: | --- |
| Trainable parameters / GFLOPs | 2.59M / 6.4 | 3.01M / 8.2 | YOLO11n is the smaller detector by parameter count and computation. |
| Fine-tuned checkpoint size | 5.21 MB | 5.96 MB | Both models are lightweight; YOLO11n produced the smaller checkpoint in this experiment. |
| Mean 20-epoch training time on RTX 3070 Ti Laptop GPU | 78.1 s | 67.2 s | YOLOv8n trained slightly faster in the recorded runs. |
| Mean test MAE across three seeds | 7.99 | 10.08 | YOLO11n had lower average detector-based counting error. |
| Seed-level paired MAE test | t = 1.82, p = 0.211 | Reference comparator | Not significant with only three paired seeds; interpret descriptively. |
| Base-identity paired absolute-error test | t = 3.21, p = 0.003; dz = 0.61 | Reference comparator | Supports YOLO11n count-error reduction after averaging related test variants by base identity. |
| Sign test over non-tied base identities | 15 better, 2 worse; p = 0.002 | Reference comparator | Conservative support for YOLO11n's counting advantage. |
| Post-processing sweep MAE | 6.15 +- 1.39 | 4.21 +- 0.98 | Validation-selected confidence/NMS settings reversed the count ranking and favored YOLOv8n. |
| Post-processing paired MAE test | Reference comparator | p = 0.025; dz = -3.56; Wilcoxon/sign p = 0.250 | The parametric test suggests a difference, but the Wilcoxon/sign test is limited by only three seeds. |

## Table 6. Reviewer Training Ablation Summary

| Training Case | YOLO11n Test MAE (mean +- std) | YOLOv8n Test MAE (mean +- std) | Interpretation |
| --- | ---: | ---: | --- |
| Default 20 epochs | 7.99 +- 1.71 | 10.08 +- 1.55 | Original controlled lightweight benchmark. |
| Default 50 epochs | 9.21 +- 1.18 | 14.33 +- 0.73 | Longer training increased mAP but worsened count MAE, suggesting count overfitting/calibration drift. |
| Minimal augmentation, 20 epochs | 4.27 +- 0.60 | 4.78 +- 0.69 | Best training ablation; reduced augmentation improved image-level count calibration. |
| Robust augmentation, 20 epochs | 11.73 +- 0.83 | 9.61 +- 1.22 | Strong augmentation did not improve counting on the current test distribution. |

## Table 7. Methodological Clarifications Added for Revision

| Reviewer Concern | Revision Added |
| --- | --- |
| Grouped split and leakage prevention | Split is grouped by base image identity extracted before augmentation suffix; related variants stay in the same split. |
| Annotation quality | Added sanity check, class-id remapping details, polygon-derived geometry caveat, and limitation on missing inter-annotator validation. |
| Epoch rationale | Added default-50-epoch ablation; longer training improved mAP but worsened count MAE. |
| Augmentation | Added exact YOLO augmentation settings and reviewer ablations for minimal and stronger augmentation. |
| Threshold optimization | Clarified validation-only confidence sweep and noted possible validation-selection bias. |
| Post-processing | Added confidence/IoU/NMS sweep with validation-selected settings; advanced soft-NMS/WBF remain future work. |
| Computational efficiency | Added parameters, GFLOPs, checkpoint size, and recorded training time; softened edge-deployment claims because CPU/edge FPS was not measured. |
