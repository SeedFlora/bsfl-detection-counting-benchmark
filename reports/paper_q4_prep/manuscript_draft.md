# Manuscript Draft - Revised After Review

Note: citation markers should be aligned with the final Jurnal RESTI reference style before submission.

## Title

Automatic Detection and Counting of Black Soldier Fly Larvae Using Classical Computer Vision and Multi-Seed Lightweight YOLO Evaluation

## Authors

Budi Juarto; Mochamad Haldi Widianto

## Abstract

This study presents a reproducible computer vision benchmark for automatic detection and counting of black soldier fly larvae (BSFL) from image data. A curated single-class detection and counting dataset was prepared from the available BSFL image resources, resulting in 321 images with 6,612 annotated larval instances split into 224 training images, 52 validation images, and 45 test images. The benchmark compares two technical pathways under a shared evaluation protocol: classical image processing and feature-based count regression, and lightweight detector-based counting using YOLO models. Classical baselines included Otsu-based connected components, adaptive thresholding, watershed segmentation, and handcrafted-feature regressors. Detector experiments used an Ultralytics YOLO workflow, followed by a strengthened multi-seed evaluation of YOLOv8n and YOLO11n.

Among the classical models, Extra Trees achieved the lowest counting MAE of 7.03, while HistGradientBoosting achieved the lowest RMSE of 14.37 and the highest classical R2 of 0.859. In the original multi-seed detector benchmark, YOLO11n achieved the best average detector-based counting performance, with test MAE of 7.99 +- 1.71, whereas YOLOv8n achieved the strongest average localization metrics, with mAP50 of 0.860 +- 0.010 and mAP50-95 of 0.697 +- 0.010. Reviewer-motivated additional experiments showed that counting accuracy was highly sensitive to training augmentation and post-processing: minimal-augmentation YOLO11n reduced MAE to 4.27 +- 0.60, while validation-selected confidence/NMS settings reduced YOLOv8n MAE to 4.21 +- 0.98. Density-aware analysis still showed that counting error increased sharply in high-density images, indicating that overlap-driven fragmentation and duplicate detections remain the main failure modes. The main contribution is therefore not a new detector architecture, but a reproducible, density-aware benchmarking framework that separates localization quality, post-processing behavior, and image-level counting reliability for BSFL monitoring.

## Keywords

Black soldier fly larvae; object detection; counting; computer vision; YOLO; reproducible benchmark

## 1. Introduction

Black soldier fly larvae (BSFL) are increasingly used in organic waste conversion, sustainable feed production, and insect-based bioprocessing. As production scales up, routine monitoring becomes more dependent on repeatable phenotyping and counting. Manual counting from tray or batch images is labor-intensive, slow, and difficult to standardize across operators. These constraints motivate automated vision-based tools that can support production monitoring without requiring destructive sampling or excessive labor.

Computer vision offers several possible routes for BSFL counting. Classical image processing is attractive because it is lightweight, transparent, and easy to deploy, but thresholding and connected-component methods are vulnerable to background variation, illumination changes, and object overlap. Deep object detectors can learn more robust visual representations, but their practical value for counting cannot be judged from localization metrics alone. A detector may achieve high mAP while still producing duplicate, fragmented, or missing boxes that directly affect image-level counts.

This distinction is important for larvae imagery because the target objects are small, elongated, and often overlapping. In dense scenes, a slight change in confidence threshold, non-maximum suppression behavior, or annotation noise can change the count substantially. Therefore, a useful BSFL benchmark should report detection quality and count accuracy separately, evaluate model stability beyond one training run, and analyze where failures occur across density levels.

This work addresses that need through an applied and reproducible benchmark for BSFL detection and counting. The contributions are:

1. A curated single-class detection and counting dataset derived from the available BSFL image and annotation resources, with group-based train-validation-test splitting to reduce leakage from related image variants.
2. A comparison of direct classical image-processing counters, feature-based classical regressors, and lightweight detector-based counters under one evaluation protocol.
3. A strengthened multi-seed YOLO evaluation that reports both localization metrics and image-level counting metrics.
4. A density-aware error analysis and qualitative success-failure review that make the main failure modes visible, especially over-counting in crowded scenes.
5. Reviewer-motivated sensitivity experiments covering longer training, augmentation ablation, confidence/NMS post-processing, and test-time visual perturbations.

The novelty is positioned as benchmark design and practical interpretation, rather than as a new model architecture. This framing is appropriate because the study evaluates existing method families but contributes a reproducible evaluation procedure, density-aware analysis, and deployment-oriented insight for BSFL monitoring.

## 2. Materials and Methods

### 2.1. Dataset Preparation

The detection and counting subset was constructed from the BSF_Larvae_v1 image collection, distributed in the project resources with a Zenodo record reference, and paired YOLO-format text annotations. Each image had one label file containing object-level annotations. The source labels used two class identifiers, 0 and 1, for the same target organism. To make the task formulation consistent, all instances were remapped into a single class, `larva`, while preserving the original object coordinates.

The source label lines contain object geometry in YOLO-style normalized coordinates. During detector training and evaluation, these annotations are used as object-level supervision and interpreted by the YOLO workflow for the single-class detection task. A light annotation sanity check was performed before export: all 6,612 label entries had valid normalized coordinates, and no malformed label line was detected. The source class-id distribution before remapping was 4,713 instances with class id 0 and 1,899 instances with class id 1. Polygon-derived bounding-box area was highly variable, with median normalized box area 0.0032 and interquartile range 0.0024 to 0.0190, which reflects the mixture of sparse and crowded images.

After curation, the final dataset contained 321 images and 6,612 annotated larvae. The split contained 224 training images, 52 validation images, and 45 test images. The corresponding object counts were 4,757 instances in training, 721 in validation, and 1,134 in testing. To reduce leakage, the split was grouped by the base image identity extracted before the augmentation suffix. Thus, image variants derived from the same base image were assigned to the same split rather than being scattered across train, validation, and test partitions.

The final dataset was exported in a YOLO-compatible directory structure with separate `images` and `labels` folders for train, validation, and test splits. A manifest file was also generated to link each image to its source path, split, base identity, and object count. The manifest was used consistently by the classical counting, detector-counting, density-analysis, and plotting scripts.

The current study does not include an independent relabeling campaign, inter-annotator agreement measurement, or manual spatial-accuracy audit of every object outline. Therefore, the annotation quality analysis should be viewed as a sanity check rather than full annotation validation. Claims about benchmark reliability are consequently limited to the curated source annotations used here.

### 2.2. Tasks, Metrics, and Statistical Analysis

Two related tasks were evaluated. The first task was object detection, where the model localized larvae instances in each image. The second task was image-level counting, where the goal was to estimate the total number of larvae in an image. For classical baselines, the predicted count was produced directly from segmentation heuristics or count-regression models. For detector-based methods, the predicted count was the number of retained detections after confidence filtering and standard YOLO post-processing.

Detection performance was evaluated using precision, recall, mAP50, and mAP50-95 on the test split. Counting performance was evaluated using mean absolute error (MAE), root mean squared error (RMSE), R2, and exact-match rate. MAE is emphasized in the interpretation because it has a direct operational meaning: the average number of larvae by which an image-level prediction is wrong.

Because the detector benchmark includes repeated seeds, results are reported as mean +- standard deviation across three seeds. In addition, paired tests were performed for the reviewer-requested comparisons: YOLOv8n versus YOLO11n, default 50 epochs versus default 20 epochs, minimal/robust augmentation versus default augmentation, and validation-selected post-processing settings. Paired t-tests are reported together with Wilcoxon or exact sign-test p-values where applicable. These tests are interpreted cautiously because seed-level comparisons contain only three pairs. To reduce dependence among augmented image variants in the original detector comparison, a second paired analysis also averaged absolute error by base image identity on the test set before comparing the two detector families.

### 2.3. Classical Counting Baselines

Three direct computer vision baselines were implemented: Otsu-based connected components, adaptive-threshold connected components, and watershed-based counting. All three methods used grayscale preprocessing and morphological cleanup, but they differed in how foreground masks and candidate larval regions were derived.

The Otsu method applied Gaussian smoothing, binary thresholding, and morphological opening and closing before counting connected components within a valid area range. The adaptive-threshold method used Gaussian adaptive thresholding and connected-component analysis with relaxed area filtering. The watershed baseline estimated foreground and background from a thresholded image, used a distance transform to define likely object centers, and applied watershed segmentation to split touching regions.

In addition, feature-based classical regressors were trained to predict image-level count directly. The extracted features included grayscale statistics, edge density, foreground fraction, Laplacian variance, several Otsu-component counts, a 16-bin grayscale histogram, and an 8 x 8 resized grayscale thumbnail. These features were evaluated with Random Forest, Extra Trees, K-nearest neighbors, and HistGradientBoosting regressors. This design provides stronger classical baselines than pure thresholding because it allows the regressor to combine multiple weak visual cues into a count estimate.

### 2.4. Detector-Based Counting with YOLO

Deep learning experiments were conducted with lightweight YOLO detectors. The initial screening stage included YOLOv8n, YOLO11n, and YOLO11s. Based on that screening, YOLOv8n and YOLO11n were selected for the strengthened multi-seed benchmark because they offered the best practical trade-off between model size and count performance.

All detector runs used 640-pixel input resolution, batch size 16, deterministic seed control, and the same curated dataset split. The original benchmark used 20 epochs as a practical lightweight budget for a moderate-sized dataset. To address the concern that this setting might underfit, an additional default-augmentation 50-epoch experiment was performed for YOLOv8n and YOLO11n across the same three seeds.

The default Ultralytics training configuration included HSV color jitter (`hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`), translation (`translate=0.1`), scale augmentation (`scale=0.5`), horizontal flipping (`fliplr=0.5`), mosaic augmentation (`mosaic=1.0`), and random erasing (`erasing=0.4`). Vertical flipping, mixup, cutmix, shear, and perspective augmentation were not enabled in the default run. Two additional 20-epoch augmentation ablations were then evaluated: a minimal-augmentation setting that disabled HSV/geometric/mosaic/mixup/erasing augmentation, and a stronger-augmentation setting that increased HSV variation, geometric perturbation, mosaic, mixup, and erasing.

For detector-based counting, the predicted image count was defined as the number of retained boxes. Because count accuracy is sensitive to confidence filtering and non-maximum suppression, two levels of post-processing evaluation were used. The original detector benchmark used a validation-only confidence sweep over thresholds 0.05, 0.10, 0.20, 0.30, and 0.40. The reviewer-response experiment expanded this into a confidence/NMS grid using confidence thresholds from 0.05 to 0.60, IoU thresholds from 0.30 to 0.80, and optional class-agnostic NMS. The configuration minimizing validation MAE was then fixed before test-set evaluation. This procedure avoids direct test-set tuning, but it can still introduce validation-set selection bias; therefore, external validation is needed before deployment claims are generalized.

Robustness was probed with test-time perturbations on the held-out test set, including brightness decrease/increase, contrast decrease/increase, Gaussian blur, Gaussian noise, and synthetic occlusion. These perturbations are not a substitute for real multi-farm external validation, but they provide a controlled sensitivity check for the kinds of imaging variation expected in production settings.

### 2.5. Computational Environment and Reproducibility

Experiments were executed with a Docker-based workflow on an NVIDIA GeForce RTX 3070 Ti Laptop GPU using CUDA-enabled PyTorch and Ultralytics. The repository contains scripts to prepare the dataset, run classical baselines, train detector variants, evaluate detector-based counting, aggregate multi-seed results, and rebuild manuscript figures and tables. The main scripts include `scripts/prepare_datasets.py`, `scripts/benchmark_detection_counting_methods.py`, `scripts/benchmark_yolo_variants.py`, `scripts/benchmark_yolo_multiseed.py`, `scripts/benchmark_yolo_postprocessing_sweep.py`, `scripts/benchmark_yolo_robustness.py`, `scripts/benchmark_yolo_reviewer_training.py`, `scripts/analyze_reviewer_experiments.py`, and `scripts/build_ieee_split_figures.py`.

YOLO11n contained 2.59M trainable parameters and required 6.4 GFLOPs, while YOLOv8n contained 3.01M parameters and required 8.2 GFLOPs. The fine-tuned checkpoint sizes were 5.21 MB for YOLO11n and 5.96 MB for YOLOv8n. Mean 20-epoch training time across three seeds was 78.1 s for YOLO11n and 67.2 s for YOLOv8n on the RTX 3070 Ti Laptop GPU. These values support the feasibility of lightweight training and evaluation, but they do not replace a full deployment benchmark. CPU latency, edge-device FPS, memory use under batch-1 inference, and quantized deployment were not measured in the current study.

## 3. Results

### 3.1. Dataset Characteristics and Annotation Sanity Check

The curated dataset comprised 321 images with 6,612 larval annotations. The test split contained 45 images and 1,134 larvae, providing a challenging final evaluation set because several images contain dense larval aggregations. The grouped split yielded 28 unique base identities in the test set, which is important because several images are related variants of the same original acquisition.

The label harmonization step was necessary because the source class identifiers were inconsistent for a single biological target. Remapping all instances to one `larva` class made the experiment consistent with the intended task: instance detection and counting rather than subtype classification. The annotation sanity check did not find malformed coordinates, but the study does not claim that all object outlines are spatially perfect. This caveat matters because noisy or inconsistent annotation boundaries can affect both mAP and count-derived conclusions, especially in crowded images.

### 3.2. Main Benchmark Comparison

Among the classical methods, the feature-based regressors were the strongest baselines. Extra Trees achieved the lowest classical counting MAE at 7.03, while HistGradientBoosting achieved the lowest classical RMSE at 14.37 and the highest classical R2 at 0.859. These results show that approximate count information is recoverable from global and local image features even without explicit object localization.

The direct segmentation-style baselines were much weaker. Watershed counting was the best of the direct image-processing methods, but its test MAE was 14.80 and R2 was 0.267. Otsu-connected components, adaptive connected components, distance-peak counting, and blob-detector counting performed worse. This confirms that simple foreground separation is insufficient when larvae overlap, vary in contrast, or form dense groups.

For detector-based counting, multi-seed evaluation changed the interpretation from a single-run ranking to a stability-aware ranking. YOLO11n achieved the best average detector-based counting result, with test MAE 7.99 +- 1.71, RMSE 19.37 +- 4.89, and R2 0.733 +- 0.128. YOLOv8n had weaker average counting results, with test MAE 10.08 +- 1.55 and R2 0.651 +- 0.129.

The localization ranking differed from the counting ranking. YOLOv8n achieved stronger average detection metrics, with precision 0.771 +- 0.008, recall 0.872 +- 0.011, mAP50 0.860 +- 0.010, and mAP50-95 0.697 +- 0.010. YOLO11n achieved precision 0.767 +- 0.014, recall 0.861 +- 0.013, mAP50 0.848 +- 0.003, and mAP50-95 0.682 +- 0.002. Thus, the detector with better mAP was not the detector with better count accuracy.

The best single detector run was YOLO11n with seed 2. It achieved precision 0.753, recall 0.876, mAP50 0.845, mAP50-95 0.682, MAE 6.07, RMSE 14.27, and R2 0.861. This single run outperformed all classical counting baselines by MAE, but the multi-seed average remains the more conservative basis for model recommendation.

### 3.3. Reviewer-Oriented Sensitivity Experiments

The longer-training ablation showed that extending training from 20 to 50 epochs did not improve image-level counting, even though localization metrics increased. For YOLO11n, default 50-epoch training increased mAP50 from 0.848 to 0.869 but worsened count MAE from 7.99 to 9.21. For YOLOv8n, default 50-epoch training kept mAP50 similar but worsened count MAE from 10.08 to 14.33. Thus, longer training did not solve the counting problem and may have increased over-counting or count-calibration mismatch.

The augmentation ablation produced the largest improvement. Minimal augmentation reduced YOLO11n MAE to 4.27 +- 0.60 and YOLOv8n MAE to 4.78 +- 0.69. Compared with default 20-epoch training, the MAE reduction was 3.72 larvae per image for YOLO11n and 5.30 larvae per image for YOLOv8n. The paired t-test for YOLOv8n was significant (p = 0.009), while the YOLO11n comparison was borderline (p = 0.051); however, Wilcoxon/sign-test p-values remained 0.250 because only three seeds were available. Stronger augmentation did not improve counting on the current test distribution.

The post-processing sweep also substantially changed the count ranking. With validation-selected confidence and NMS settings, YOLOv8n achieved MAE 4.21 +- 0.98 and RMSE 9.62 +- 3.21, while YOLO11n achieved MAE 6.15 +- 1.39 and RMSE 16.14 +- 4.74. The paired t-test across seeds favored YOLOv8n after post-processing (p = 0.025 for MAE), although the non-parametric result remained limited by the three-seed sample. This result reinforces that detector-based BSFL counting depends not only on the detector backbone but also on confidence and NMS calibration.

The test-time robustness sensitivity analysis showed that YOLO11n generally had lower MAE than YOLOv8n under brightness, contrast, Gaussian noise, and synthetic-occlusion perturbations. The main exception was Gaussian blur, where YOLOv8n improved to MAE 4.33 while YOLO11n reached MAE 7.53. These perturbation results should be interpreted as controlled sensitivity checks, not as proof of field robustness across farms or camera systems.

### 3.4. Statistical Comparison of Detector Count Errors

At the seed level, YOLO11n had lower MAE than YOLOv8n on average by 2.09 larvae per image. However, with only three paired seeds, the paired t-test was not statistically significant (t = 1.82, p = 0.211), and the result should be treated as descriptive.

A second paired analysis averaged absolute error by test base identity to reduce dependence from related image variants. Across 28 base identities, YOLO11n reduced absolute error relative to YOLOv8n by an average of 3.13 larvae per base identity. The paired t-test was significant (t = 3.21, p = 0.003), with a moderate standardized paired effect size (Cohen's dz = 0.61). A conservative sign test over non-tied base identities also favored YOLO11n: it was better in 15 cases and worse in 2 cases (two-sided p = 0.002). These post-hoc statistics support the observed counting advantage of YOLO11n, although they should be interpreted cautiously because the dataset is still small.

The additional sensitivity experiments add a more nuanced interpretation. Under default training and a fixed confidence-selection protocol, YOLO11n counted better than YOLOv8n. After confidence/NMS post-processing optimization, YOLOv8n counted better. After training with minimal augmentation, both detectors improved sharply and YOLO11n again had the lower average MAE. Therefore, the statistically supported conclusion is not that one detector is universally superior, but that BSFL counting must be evaluated as a coupled detector-training and post-processing problem.

### 3.5. Detection-Versus-Counting Trade-Off

The central scientific finding is that localization quality and counting reliability are related but not identical. YOLOv8n produced better mAP values, yet YOLO11n produced better image-level count accuracy. This can occur because mAP rewards overlap quality and ranking across detections, whereas counting penalizes the final number of retained boxes. A detector may improve localization of visible larval parts while also producing extra boxes on overlapping body regions, and those extra boxes directly increase count error.

YOLO11n's advantage appears to be count calibration rather than universally better localization. Its lower count MAE suggests that, at the selected confidence threshold, it produced fewer costly over-counting errors in dense images. YOLOv8n, by contrast, retained stronger localization metrics but was more vulnerable to count inflation in several high-density cases. This distinction is operationally important because a farm-monitoring system usually needs reliable counts, while a visual inspection tool may prioritize box placement and recall.

### 3.6. Density-Aware Error Analysis

Counting error increased with image density. For low-density images, average MAE was 2.38 for YOLO11n and 2.60 for YOLOv8n. For medium-density images, MAE increased to 5.04 and 6.38, respectively. For high-density images, error increased sharply to 16.56 for YOLO11n and 21.27 for YOLOv8n.

This pattern indicates that sparse-scene counting is already relatively reliable, while crowded-scene counting remains the main challenge. In high-density images, overlapping larvae can cause one larva to be split into multiple detections, nearby larvae to suppress each other, or ambiguous partial body regions to be counted as separate objects. These errors are not merely numerical; they affect real production use because high-density trays are precisely where manual counting is most difficult and automated monitoring would be most valuable.

### 3.7. Visual Evaluation

The benchmark figures support the numerical results. The multi-seed bar plots show that YOLOv8n has stronger average mAP, whereas YOLO11n has lower average count MAE. Scatter plots show that both the best classical regressor and the best YOLO11n run follow the general count trend, but extreme crowded images produce the largest deviations. Density plots show a clear rise in error from low to high density.

The qualitative success-failure montage provides an important diagnostic complement to the aggregate metrics. Exact or near-exact examples usually occur when larvae are separated enough for each instance to be represented once. Severe failures are concentrated in crowded regions, where repeated detections on overlapping larval bodies produce large positive count errors. This supports the interpretation that future improvement should focus on crowd-aware post-processing, instance segmentation, or density-estimation alternatives rather than only replacing one lightweight detector with another.

## 4. Discussion

### 4.1. Interpretation of the Main Results

The results show that BSFL counting is feasible with both classical and detector-based methods, but the best method depends on the deployment objective. Feature-based regressors are strong if only an aggregate count is needed and localization is not required. Extra Trees and HistGradientBoosting achieved competitive count accuracy while avoiding detector training complexity. However, these regressors cannot show where larvae were counted, which limits their usefulness for visual verification, quality control, and downstream spatial analysis.

Detector-based counting provides richer output because each count is linked to localized predictions. In the original default-training benchmark, the best YOLO11n run achieved the lowest single-run MAE and YOLO11n had the best average detector-based count accuracy across seeds. Nevertheless, YOLOv8n retained stronger detection metrics. The reviewer-response experiments further showed that this ranking is sensitive to the training and post-processing protocol: minimal augmentation made both detectors substantially better, while the confidence/NMS sweep made YOLOv8n the strongest post-processed counter. Model selection should therefore be based on the operational endpoint and the full counting pipeline, not only the detector family.

### 4.2. Why Better mAP Did Not Guarantee Better Counting

The divergence between mAP and MAE is one of the most important findings of the study. mAP evaluates whether predicted boxes overlap annotated objects at different confidence levels. Counting evaluates the final number of boxes retained after confidence filtering and NMS. These objectives can diverge in crowded larvae images because one extra high-confidence duplicate box can have little effect on mAP ranking but increases count error by one. Repeated duplicate boxes in dense areas can produce large over-counting even when localization metrics remain acceptable.

This also explains why post-processing deserves more attention. Standard NMS is designed to remove highly overlapping boxes, but elongated larvae can overlap partially, cross each other, or be visible only in fragments. In those cases, a fixed IoU threshold may fail to remove duplicates or may suppress true neighboring larvae. The validation-selected confidence/NMS sweep reduced YOLOv8n MAE from 10.08 to 4.21, demonstrating that a simple post-processing calibration can be as influential as changing the detector. More advanced variants such as soft-NMS, weighted box fusion, density-adaptive thresholds, or morphology-informed duplicate merging remain promising follow-up work.

### 4.3. Practical Implications for BSFL Production

For production monitoring, the density-wise results are more actionable than a single overall MAE. Low-density scenes can already be counted with small errors, but high-density scenes remain difficult. This means the method may be useful for routine low-to-medium density inspection, while dense tray images require caution. A practical deployment should flag high-density images for manual review, combine detector counts with uncertainty indicators, or use density-specific thresholds.

The computational results support lightweight feasibility, but only within the tested GPU environment. The fine-tuned YOLO checkpoints are small, and 20-epoch training took less than two minutes per run on the RTX 3070 Ti Laptop GPU. The perturbation analysis suggests that YOLO11n is generally more stable under brightness, contrast, noise, and synthetic occlusion changes, while YOLOv8n is unusually strong under blur after the tested perturbation. However, real deployment requires additional measurement of batch-1 latency, CPU performance, memory use, quantized inference, camera resolution effects, and true cross-farm illumination shifts. The present results should therefore be interpreted as a benchmark toward deployability, not a complete edge-device validation.

### 4.4. Limitations and Future Work

Several limitations remain. First, the test set contains only 45 images and 28 base identities, so the benchmark should be viewed as promising but not definitive. Second, the annotation quality check did not include independent relabeling, inter-annotator agreement, or systematic spatial-error measurement. This weakens any claim that the benchmark fully captures ground-truth uncertainty, especially for small or overlapping larvae.

Third, the study compares classical counting and bounding-box detector-based counting, but it does not benchmark other method families that may be better suited to crowded scenes. Future work should evaluate instance segmentation models such as Mask R-CNN or YOLO-seg, density-map CNNs, point-supervised crowd-counting approaches, and tracking-by-detection for video-based monitoring. These alternatives may better handle overlap, but they also introduce additional annotation, training, and deployment costs.

Fourth, the augmentation and robustness experiments are still sensitivity analyses on the current dataset rather than external validation. Minimal augmentation improved counting strongly, while stronger augmentation did not, suggesting that aggressive transformations may distort count calibration for this dataset. Future experiments should test these settings on independently collected farm images with different trays, lighting, camera distances, and larval stages. Fifth, confidence and NMS settings were optimized on the validation set only; a larger external validation set is needed to ensure that threshold selection does not overfit the current data distribution.

Finally, the current post-processing analysis is broader than the original benchmark but still limited to confidence, IoU, and class-agnostic NMS choices. The results suggest that crowded-scene over-counting is a major error source, but soft-NMS, weighted box fusion, density-adaptive duplicate removal, and segmentation-aware post-processing were not tested. These are low-cost improvements that should be evaluated before concluding that a new detector architecture is required.

## 5. Conclusion

This study developed a reproducible benchmark for automatic detection and counting of black soldier fly larvae using classical computer vision and lightweight YOLO detectors. The benchmark shows that automatic counting is feasible on the curated dataset. Among classical methods, Extra Trees achieved the lowest MAE and HistGradientBoosting achieved the strongest RMSE and R2. Among detector-based methods, YOLO11n achieved the strongest average counting accuracy, while YOLOv8n achieved the strongest average detection metrics.

The study's main finding is that detection quality and counting quality should be evaluated separately. YOLOv8n had stronger localization metrics, YOLO11n had better default counting accuracy, and validation-selected post-processing could reverse the count ranking. The best additional results were achieved by minimal-augmentation YOLO11n (MAE 4.27 +- 0.60) and post-processed YOLOv8n (MAE 4.21 +- 0.98). High-density images remained the dominant failure mode for both detectors. The work therefore contributes a reproducible and density-aware evaluation framework for BSFL counting, while identifying annotation validation, advanced post-processing, instance segmentation, density-map models, and external deployment testing as priorities for future research.

## Suggested Figure and Table Placement

Table 1 should present the curated dataset summary. Table 2 should report the multi-seed detector comparison. Table 3 should compare the strongest classical, detector, and reviewer-ablation counting results. Table 4 should report density-wise counting error. Table 5 should report computational and statistical addenda, including checkpoint size, training time, parameters, GFLOPs, and paired count-error tests. Table 6 should summarize the reviewer training ablations.

Figure 1 should show the experimental workflow. Figure 2 should show the detector-based counting protocol. Figure 3 should show the dataset split and count distribution. Figures 4 and 5 should compare classical and detector-based metrics. Figure 6 should show density-wise counting error. Figure 7 should show predicted versus ground-truth counts for the strongest classical and detector-based methods. Figure 8 should show qualitative success and failure examples with captions that explicitly state the error pattern illustrated.
