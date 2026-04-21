# Draft Abstract

## Suggested Title

Automatic Detection and Counting of Black Soldier Fly Larvae Using Classical Computer Vision and Multi-Seed Lightweight YOLO Evaluation

## Abstract

This study presents a reproducible computer vision pipeline for automatic detection and counting of black soldier fly larvae from image data. A curated detection dataset was prepared from the available project resources, resulting in 321 images with 6,612 annotated larvae, split into 224 training images, 52 validation images, and 45 test images. We compared classical counting approaches and lightweight deep learning detectors under a shared evaluation protocol. Classical baselines included Otsu-based connected components, adaptive thresholding, watershed segmentation, and several handcrafted-feature regressors. Detector-based experiments were conducted using GPU-enabled Docker workflows. After an initial model screening, the two strongest lightweight detectors, YOLOv8n and YOLO11n, were further evaluated with three random seeds each.

Among the classical models, Extra Trees achieved the lowest counting MAE of 7.03, while HistGradientBoosting achieved the lowest RMSE of 14.37 and the highest classical R2 of 0.859. In the strengthened multi-seed detector benchmark, YOLO11n achieved the best average detector-based counting performance, with test MAE of 7.99 +- 1.71 and test R2 of 0.733 +- 0.128, while YOLOv8n achieved the strongest average detection quality, with mAP50 of 0.860 +- 0.010 and mAP50-95 of 0.697 +- 0.010. The best single deep-learning run was YOLO11n with seed 2, reaching MAE 6.07 and R2 0.861, outperforming all classical counting baselines. Density-aware analysis further showed that counting error increased substantially in crowded scenes, although YOLO11n remained more accurate than YOLOv8n across low-, medium-, and high-density bins. These results indicate that automatic larvae counting is feasible with moderate-sized datasets and that lightweight detector-based pipelines can provide strong practical performance when validated beyond a single run. Future work should focus on external validation, broader architectures, and larger datasets.
