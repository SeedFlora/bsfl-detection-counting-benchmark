# Manuscript Draft

Note: bracketed citation placeholders should be replaced with journal-appropriate references before submission.

## Title

Automatic Detection and Counting of Black Soldier Fly Larvae Using Classical Computer Vision and Multi-Seed Lightweight YOLO Evaluation

## Abstract

This study presents a reproducible computer vision pipeline for automatic detection and counting of black soldier fly larvae from image data. A curated detection dataset was prepared from the available project resources, resulting in 321 images with 6,612 annotated larvae, split into 224 training images, 52 validation images, and 45 test images. We compared classical counting approaches and lightweight deep learning detectors under a shared evaluation protocol. Classical baselines included Otsu-based connected components, adaptive thresholding, watershed segmentation, and several handcrafted-feature regressors. Detector-based experiments were conducted using GPU-enabled Docker workflows. After an initial model screening, the two strongest lightweight detectors, YOLOv8n and YOLO11n, were further evaluated with three random seeds each.

Among the classical models, Extra Trees achieved the lowest counting MAE of 7.03, while HistGradientBoosting achieved the lowest RMSE of 14.37 and the highest classical R2 of 0.859. In the strengthened multi-seed detector benchmark, YOLO11n achieved the best average detector-based counting performance, with test MAE of 7.99 +- 1.71 and test R2 of 0.733 +- 0.128, while YOLOv8n achieved the strongest average detection quality, with mAP50 of 0.860 +- 0.010 and mAP50-95 of 0.697 +- 0.010. The best single deep-learning run was YOLO11n with seed 2, reaching MAE 6.07 and R2 0.861, outperforming all classical counting baselines. Density-aware analysis further showed that counting error increased substantially in crowded scenes, although YOLO11n remained more accurate than YOLOv8n across low-, medium-, and high-density bins. These results indicate that automatic larvae counting is feasible with moderate-sized datasets and that lightweight detector-based pipelines can provide strong practical performance when validated beyond a single run. Future work should focus on external validation, broader architectures, and larger datasets.

## Keywords

Black soldier fly larvae; object detection; counting; computer vision; YOLO; reproducible benchmark

## 1. Introduction

Black soldier fly larvae are increasingly important in waste conversion, sustainable feed production, and insect-based bioprocessing [insert refs]. As the scale of larvae production increases, routine monitoring becomes more demanding and more dependent on accurate phenotyping and counting. Manual counting from tray or batch images is labor-intensive, time-consuming, and difficult to standardize across operators. These limitations motivate the development of automated vision-based tools for larvae analysis.

Computer vision offers a practical pathway for replacing or assisting manual counting in biological production settings. Classical image processing methods remain attractive because they are lightweight, interpretable, and easy to deploy. However, these approaches often struggle when larvae overlap, when backgrounds vary, or when object density becomes high. In contrast, deep learning detectors can learn more robust visual representations, but their performance on moderate-sized biological datasets is not always guaranteed and their deployment cost can be higher [insert refs].

Several recent studies have shown that detector-based counting can be a practical compromise between instance localization and aggregate quantification [insert refs]. For insect imagery and agricultural monitoring tasks, a key question is not only whether a detector achieves high localization metrics, but also whether the resulting detections translate into reliable count estimates. This distinction matters because a model may produce acceptable mean average precision while still over-counting or under-counting in dense scenes.

This work presents an applied and reproducible benchmark for black soldier fly larvae detection and counting using curated project data. The main contributions are threefold. First, we prepared a consistent detection and counting dataset from the available image and label resources, including label harmonization into a single larva class and a stable train-validation-test split. Second, we compared multiple classical counting baselines with lightweight detector-based approaches under the same evaluation protocol. Third, we combined initial detector screening with strengthened multi-seed evaluation and density-aware error analysis, allowing the paper to report not only performance but also stability and failure behavior under crowded scenes.

## 2. Materials and Methods

### 2.1. Dataset preparation

The data used in this study were derived from the project resources available in the local workspace. The detection and counting subset was constructed from the BSF larvae image collection and its paired text annotations. Each image was associated with a label file containing object-level annotations. During curation, the original class identifiers were inspected and found to be inconsistent across part of the source dataset. To stabilize the task formulation, all object annotations were remapped to a single class, namely `larva`, while preserving the original bounding box geometry.

After curation, the final detection dataset contained 321 images with 6,612 annotated larvae. The dataset was partitioned into 224 training images, 52 validation images, and 45 test images. The corresponding object counts were 4,757 instances in training, 721 instances in validation, and 1,134 instances in testing. Group-based splitting was used to reduce leakage from near-duplicate images derived from the same base image identity. This produced a more conservative and reproducible evaluation setting than a purely random split.

The final dataset was exported in YOLO-compatible directory structure with separate `images` and `labels` folders for each split and a dataset YAML file describing the single-class detection problem. In addition, a manifest file was created to link each image to its object count and split membership so that count-based and detector-based experiments could be evaluated consistently.

### 2.2. Experimental tasks and evaluation metrics

Two related tasks were considered. The first task was object detection, where the goal was to localize larvae instances in each image. The second task was image-level counting, where the goal was to estimate the number of larvae in each image. For classical counting baselines, the count was predicted directly from image processing or regression outputs. For detector-based approaches, the count was computed as the number of predicted bounding boxes after confidence selection.

Detection performance was evaluated on the test split using precision, recall, mAP50, and mAP50-95. Counting performance was evaluated using mean absolute error, root mean squared error, and R2. In addition, exact-match rate was tracked internally to measure the fraction of test images where the predicted count exactly matched the true count. Because counting quality was a major practical objective of the study, model comparison emphasized both detection and counting metrics rather than detection scores alone.

### 2.3. Classical counting baselines

Three direct computer vision baselines were implemented: Otsu-based connected components, adaptive-threshold connected components, and watershed-based counting. All three methods used grayscale preprocessing and morphological cleanup, but they differed in how the foreground mask was estimated and how candidate object regions were separated.

The Otsu-based method applied Gaussian smoothing, binary thresholding, and morphological opening and closing before counting connected components within a valid area range. The adaptive-threshold method used Gaussian adaptive thresholding and morphological opening to derive a foreground mask, then applied connected-component analysis with a relaxed area threshold. The watershed baseline estimated foreground and background regions from a thresholded image, used a distance transform to define sure foreground seeds, and applied watershed segmentation to split touching larvae before counting segmented regions that satisfied area constraints.

In addition to these direct counting baselines, a family of classical machine learning models was built to predict count from handcrafted image features. For each image, grayscale statistics, edge density, foreground fraction, Laplacian variance, multiple Otsu-based component counts, a 16-bin grayscale histogram, and an 8x8 resized grayscale thumbnail were extracted. These features were then evaluated with random forest, Extra Trees, K-nearest neighbors, and HistGradientBoosting regressors. This feature-based design represents a stronger classical alternative than pure thresholding because it can combine multiple weak visual signals into a direct count predictor.

### 2.4. Detector-based counting with YOLO

Deep learning experiments were conducted with lightweight YOLO detectors. The initial screening stage included YOLOv8n, YOLO11n, and YOLO11s. Based on that first-stage comparison, YOLOv8n and YOLO11n were selected as the strongest practical candidates and were then reevaluated using three random seeds each. All detector runs were trained on the curated single-class dataset using 640-pixel input resolution, batch size 16, and 20 epochs. Training was performed with Ultralytics-based workflows, and results were stored automatically in per-run report directories.

For detector-based counting, the predicted count for an image was defined as the number of retained bounding boxes. Because counting quality is sensitive to the confidence threshold, a small confidence sweep was performed on the validation set using thresholds of 0.05, 0.10, 0.20, 0.30, and 0.40. The threshold that minimized validation MAE was then used for test-set counting evaluation. In the final experiments, the best threshold for the evaluated YOLO models was 0.40.

To compare multiple YOLO variants consistently, automated benchmark scripts were created to train each model, evaluate detector quality on the test set, derive count predictions from the trained weights, and generate comparison plots. The strengthened stage also aggregated mean and standard deviation across seeds, computed density-wise error statistics over low-, medium-, and high-density test subsets, and generated qualitative success-failure montages from the best deep-learning run. This ensured that training curves, detector metrics, count metrics, and error analyses were collected under the same reproducible procedure.

### 2.5. Computational environment and reproducibility

The experiments were executed in a Docker-based GPU workflow. The GPU environment used an NVIDIA GeForce RTX 3070 Ti Laptop GPU with CUDA-enabled PyTorch and Ultralytics. The project workspace included dedicated CPU and GPU Docker configurations, path remapping utilities for container-safe dataset access, and scripts to regenerate datasets, rerun baselines, retrain YOLO models, and rebuild paper-preparation tables. This design was intended to improve portability and reproducibility, both of which are often underreported in applied computer vision papers.

## 3. Results

### 3.1. Dataset characteristics

The curated detection dataset comprised 321 images with 6,612 object annotations. The test split alone contained 45 images and 1,134 larvae instances, providing a non-trivial basis for final evaluation. Although the dataset is moderate in size by deep learning standards, it is sufficient to examine whether lightweight detectors and classical methods can produce practically useful counting performance on black soldier fly larvae imagery.

The class space was intentionally reduced to a single `larva` category after harmonization of the original annotation files. This simplification was appropriate for the present study because the target application was instance detection and counting rather than fine-grained larval subtype recognition. It also reduced label noise introduced by inconsistent class identifiers in the source files.

### 3.2. Main benchmark comparison

Table 2 summarizes the strengthened results. Among the classical methods, the feature-based regressors were clearly strongest. Extra Trees achieved the lowest classical counting MAE at 7.03, while HistGradientBoosting achieved the lowest classical RMSE at 14.37 and the highest classical R2 at 0.859. These results confirm that image-level handcrafted features contain strong predictive information for approximate count estimation when the task is formulated directly as regression.

For deep learning, the strengthened evaluation changed the ranking observed from the initial single-run screening. After repeating training with three random seeds, YOLO11n emerged as the strongest detector-counter on average, achieving a test count MAE of 7.99 +- 1.71, test RMSE of 19.37 +- 4.89, and test R2 of 0.733 +- 0.128. By contrast, YOLOv8n achieved a worse average counting result, with test MAE of 10.08 +- 1.55 and test R2 of 0.651 +- 0.129.

The detection ranking, however, favored YOLOv8n. Across three seeds, YOLOv8n achieved mean test precision 0.771 +- 0.008, recall 0.872 +- 0.011, mAP50 0.860 +- 0.010, and mAP50-95 0.697 +- 0.010. YOLO11n was slightly weaker on average localization quality, with precision 0.767 +- 0.014, recall 0.861 +- 0.013, mAP50 0.848 +- 0.003, and mAP50-95 0.682 +- 0.002. This indicates that the strongest detector in terms of localization was not necessarily the strongest counter in terms of image-level count error.

The best single deep-learning run was obtained by YOLO11n with seed 2. That run achieved test precision 0.753, recall 0.876, mAP50 0.845, mAP50-95 0.682, counting MAE 6.07, RMSE 14.27, and R2 0.861. Notably, this counting result surpassed all classical baselines, showing that detector-based counting can outperform handcrafted regression when the run is favorable and sufficiently stable.

The direct image processing baselines remained clearly weaker than both the feature-based regressors and the detector-based models. Watershed counting was the best among the direct segmentation-style baselines, but it still reached only test MAE 14.80 and R2 0.267. Otsu-based connected components, adaptive connected components, distance-peak counting, and blob-detector counting were substantially worse. These results reinforce the difficulty of counting larvae using thresholding or peak-based heuristics alone when objects overlap and background variation is present.

### 3.3. Detection-versus-counting trade-off

An important outcome of this study is that the model ranking depends on how performance is defined. If the target objective is pure classical counting, Extra Trees provides the lowest MAE while HistGradientBoosting provides the strongest RMSE and R2. If the target objective is the best single overall count result, the best YOLO11n run exceeds all classical baselines. If the target objective is average detector-based counting across repeated runs, YOLO11n is the best-performing lightweight detector in the current benchmark. If the target objective is average localization quality, YOLOv8n is the strongest model.

This ranking split is scientifically useful because it shows that count quality and localization quality do not always move together. YOLOv8n produced the best mean mAP values, yet YOLO11n produced the best mean counting error. Therefore, detector-based counting papers should report count-specific metrics directly rather than assuming that better average precision automatically implies better counting accuracy.

The earlier exploratory screening with YOLO11s also supports this point. In the initial benchmark, YOLO11s obtained competitive localization metrics but clearly worse counting behavior than the lighter variants. Together, the screening and multi-seed results indicate that counting-through-detection must be evaluated as its own target problem, not only as a by-product of object detection.

### 3.4. Density-aware error analysis

Density-aware evaluation showed that counting difficulty increased markedly with object density. For low-density images, average MAE was 2.38 for YOLO11n and 2.60 for YOLOv8n. For medium-density images, the corresponding MAE values increased to 5.04 and 6.38. For high-density images, the error rose much more sharply, reaching 16.56 for YOLO11n and 21.27 for YOLOv8n.

This pattern shows that the main failure mode of the current models is not sparse-scene counting, but crowded-scene counting. Even though both models remained usable, the degradation under dense scenes was substantial enough to merit explicit discussion in the paper. Importantly, YOLO11n remained more accurate than YOLOv8n in all three density bins, which strengthens the claim that its advantage is not limited to only one subset of the test distribution.

### 3.5. Visual evaluation

The generated benchmark figures support the numerical trends. The multi-seed comparison plots show that YOLO11n has the better average counting behavior, whereas YOLOv8n retains a modest advantage in average mAP-based detection quality. The density-wise plots show a consistent increase in error from low-density to high-density scenes for both models, with a steeper rise for YOLOv8n. The qualitative montage generated from the best run illustrates both successful predictions with exact counts and severe over-counting failures in crowded images.

These qualitative observations are consistent with the summary metrics. In particular, several failure examples reveal large positive errors where the predicted number of boxes exceeds the ground-truth count by more than 60 instances. Such cases are concentrated in dense scenes, reinforcing the interpretation that overlap and crowding remain the central challenge for the current pipeline.

## 4. Discussion

The strengthened results show that larvae counting is already feasible with both classical and deep learning methods, but the preferred model depends on the intended use case. The feature-based regressors remain important baselines because they deliver strong pure counting accuracy with relatively low modeling complexity. In particular, Extra Trees and HistGradientBoosting both outperformed the earlier random-forest baseline on at least one major count metric. However, the best YOLO11n run surpassed all of these classical baselines, and YOLO11n also provided the best average detector-based counting performance across repeated runs. This substantially improves the practical and scientific value of the detector-based approach.

At the same time, YOLOv8n retained the strongest average localization quality. This means the choice between YOLO11n and YOLOv8n is not binary but application-dependent. If the downstream system prioritizes accurate counts, YOLO11n is the current best candidate. If the system prioritizes detector quality and visual localization consistency, YOLOv8n remains highly attractive. For a paper, this split is useful because it reveals a non-trivial trade-off rather than a one-dimensional ranking.

The multi-seed protocol also improves the credibility of the study. Earlier, the deep-learning claims relied mainly on single-run evidence. After repeated runs, the relative advantage of YOLO11n in counting remained visible, while YOLOv8n remained stronger in detection. This makes the final narrative more robust and better aligned with journal expectations for experimental reliability.

The density-aware analysis sharpens the interpretation further. Both models degraded as the scenes became more crowded, and the worst failures were concentrated in high-density images. This suggests that the main bottleneck is not whether larvae are visually detectable in general, but whether the detector can avoid duplicate or fragmented predictions when many larvae overlap. That observation provides a concrete motivation for future work on crowd-aware training, denser annotations, or post-processing strategies tuned for biological aggregation scenes.

Several limitations still remain. First, the test set contains only 45 images, so the benchmark should still be viewed as promising rather than definitive. Second, the current repeated evaluation covers only two deep models and three seeds each, leaving room for broader architectural comparison. Third, the density-wise R2 values are unstable in the low- and medium-density bins because the within-bin target variance is small, so MAE is the more reliable interpretation in those subsets. Despite these limitations, the study now includes multi-seed evaluation, density-aware error analysis, and qualitative failure visualization, which together make the paper substantially stronger than a single-run benchmark.

## 5. Conclusion

This study developed a reproducible benchmark for black soldier fly larvae detection and counting using curated project data. The results show that automatic counting is practical on the current dataset. The strongest classical counting baselines were Extra Trees for MAE and HistGradientBoosting for RMSE and R2, while the strengthened detector benchmark showed that YOLO11n was the strongest model for average counting accuracy and YOLOv8n was the strongest model for average detection quality. The best single deep-learning run, obtained by YOLO11n seed 2, achieved counting accuracy that exceeded all classical baselines.

Overall, the findings support a clear recommendation for manuscript positioning: the paper should emphasize detection and counting as the primary contribution, with regression retained as an important baseline rather than as the main focus. The current version is already suitable for a Q4-style applied study, and the most valuable remaining additions would be literature-grounded framing, broader external validation, and, if feasible, one more round of model expansion beyond the current lightweight YOLO variants.

## Suggested Figure and Table Placement

Table 1 should present the curated dataset summary. Table 2 should report the multi-seed detector comparison, and Table 3 should compare the strongest classical and deep-learning counting results. Figure 1 should compare multi-seed detection metrics, Figure 2 should compare multi-seed counting metrics, Figure 3 should report density-aware counting error, and Figure 4 should show selected success and failure examples from the best deep-learning run.
