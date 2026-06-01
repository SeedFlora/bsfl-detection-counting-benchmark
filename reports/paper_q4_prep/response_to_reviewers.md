# Response to Reviewers

Manuscript title: Automatic Detection and Counting of Black Soldier Fly Larvae Using Classical Computer Vision and Multi-Seed Lightweight YOLO Evaluation

Authors: Budi Juarto; Mochamad Haldi Widianto

We thank the editor and reviewers for the constructive comments. The manuscript has been revised to improve methodological clarity, scientific interpretation, visual explanation, limitation statements, and reproducibility details. The revision also softens claims that were too broad for the current dataset size and reframes the contribution as a reproducible, density-aware benchmark rather than a new detection architecture.

## Summary of Major Revisions

1. The Introduction now states the central novelty more explicitly: a reproducible benchmark that separates localization quality from image-level counting reliability.
2. The Materials and Methods section now explains grouped splitting by base image identity, label harmonization, annotation sanity checks, YOLO augmentation settings, 20-epoch rationale, confidence/NMS selection, robustness perturbations, and post-processing limitations.
3. New reviewer-response experiments were added: confidence/NMS post-processing sweep, test-time robustness perturbations, 50-epoch training, minimal-augmentation training, and stronger-augmentation training.
4. The Results section now includes deeper interpretation of why detection mAP and count MAE diverge, including the finding that post-processing can reverse the YOLO11n-versus-YOLOv8n count ranking.
5. Statistical comparisons were added for detector count errors, training ablations, and post-processing settings.
6. Computational details were added, including parameter count, GFLOPs, checkpoint size, and recorded training time.
7. The Discussion now includes explicit limitations on annotation quality, alternative method classes, external validation, advanced post-processing, and edge deployment.
8. Figure captions were expanded to make the purpose and interpretation of the visual results clearer.

## Reviewer B

Comment: The manuscript lacks in-depth analysis in the Discussion and Results sections, and the scientific reasoning and interpretation are insufficient.

Response: We revised the Results and Discussion sections to provide deeper interpretation rather than only restating metrics. The revision now explains the difference between localization metrics and count accuracy, why YOLO11n can achieve lower MAE despite YOLOv8n having higher mAP, and how overlapping larvae lead to fragmentation and duplicate detections. We also added a density-aware interpretation that links high-density failures to practical BSFL production monitoring.

Comment: Visuals were not used appropriately or clearly.

Response: We expanded the figure captions to clarify what each visual demonstrates, especially the multi-seed mAP/counting trade-off, density-wise error growth, training convergence, and qualitative success-failure examples. The revised captions are provided in `reports/paper_q4_prep/ieee_figures_split/figure_captions_ieee_split.md`.

Comment: The original contribution was unclear.

Response: We clarified that the manuscript does not claim a new detector architecture. The contribution is the reproducible benchmark design, the separation of detection and counting metrics, multi-seed evaluation, and density-aware failure analysis for BSFL monitoring.

## Reviewer C

Comment: The manuscript needs improvement in English academic writing and contains overly long paragraphs.

Response: The manuscript was rewritten for clearer academic flow. Several long paragraphs were split, and the Introduction, Results, and Discussion were reorganized into more focused subsections.

Comment: Section numbering and structure need consistency.

Response: The revised manuscript now follows a consistent structure: Introduction, Materials and Methods, Results, Discussion, and Conclusion. The methods subsections are numbered sequentially from 2.1 to 2.5.

Comment: Figure captions and visual interpretability should be improved.

Response: The figure captions were rewritten to describe not only what is shown but also why each figure matters. The captions now explicitly identify the density-failure pattern, the detection-versus-counting trade-off, and the implication of qualitative failures.

Comment: Grouped splitting should be explained more clearly.

Response: Section 2.1 now explains that the split was grouped by base image identity extracted before the augmentation suffix, so image variants derived from the same base image remain in the same split. This reduces leakage across train, validation, and test partitions.

Comment: The 20-epoch setting should be justified.

Response: Section 2.4 and Section 3.3 now add a direct 50-epoch ablation across YOLOv8n and YOLO11n with three seeds. Longer training improved or preserved localization metrics but worsened count MAE: YOLO11n changed from 7.99 +- 1.71 to 9.21 +- 1.18 MAE, while YOLOv8n changed from 10.08 +- 1.55 to 14.33 +- 0.73 MAE. We therefore retain 20 epochs as a controlled lightweight benchmark and explain that more training did not improve the counting endpoint in this dataset.

Comment: Augmentation, preprocessing, and hyperparameters should be reported in more detail.

Response: Section 2.4 now reports the main YOLO training settings: image size, batch size, epochs, deterministic seed control, HSV color jitter, translation, scale augmentation, horizontal flipping, mosaic augmentation, and random erasing. Disabled augmentations are also identified. We also added two training ablations: minimal augmentation and stronger augmentation. Minimal augmentation produced the best training-ablation counts (YOLO11n MAE 4.27 +- 0.60; YOLOv8n MAE 4.78 +- 0.69).

Comment: Confidence-threshold optimization and possible validation bias should be explained.

Response: Section 2.4 now describes the validation-only confidence sweep over thresholds 0.05, 0.10, 0.20, 0.30, and 0.40 used in the original benchmark. The reviewer-response experiment expands this to confidence thresholds from 0.05 to 0.60, IoU thresholds from 0.30 to 0.80, and optional class-agnostic NMS. The revised text explicitly notes that validation-based selection may introduce selection bias and that external validation is needed.

Comment: Code availability and reproducibility should be explicit.

Response: Section 2.5 now lists the main scripts used to regenerate the dataset, benchmarks, multi-seed results, and figures.

Comment: The Discussion should elaborate on YOLO11n versus YOLOv8n, overlapping larvae, density-related failure, and computational trade-offs.

Response: Sections 4.1 to 4.3 now provide this interpretation. The revised manuscript explains that mAP and count accuracy can diverge because mAP evaluates localization ranking while counting penalizes the final number of retained detections.

Comment: The study should discuss additional metrics and dataset limitations.

Response: Exact-match rate, robustness perturbation results, post-processing sweep results, statistical comparison, parameter count, GFLOPs, checkpoint size, and recorded training time were added. Dataset limitations are now discussed explicitly, including the small test set and limited number of base identities.

Comment: Novelty should be framed more carefully.

Response: The Introduction and Conclusion now frame the contribution as a benchmark and analysis contribution rather than a new architecture.

Comment: References and citation formatting should be standardized.

Response: The revised draft includes a note that citation markers should be aligned with the final Jurnal RESTI style. Duplicate reference cleanup should be performed in the final formatted manuscript file because the local Markdown draft does not contain the complete submitted reference list.

## Reviewer D

Comment 1: Annotation quality is not discussed in detail.

Response: Section 2.1 now adds an annotation sanity check. The revision reports the original class-id inconsistency, remapping into one larva class, valid coordinate checks for 6,612 label entries, and polygon-derived bounding-box area statistics. The Discussion now explicitly states that no independent relabeling, inter-annotator agreement, or full spatial-accuracy audit was performed, so generalization claims are limited.

Comment 2: The benchmark does not include instance segmentation, density-map CNNs, or tracking-by-detection.

Response: Section 4.4 now explicitly lists these method families as important future work and clarifies that the current study is limited to classical counting and detector-based counting. The conclusions were softened accordingly.

Comment 3: Data augmentation and robustness are not analyzed.

Response: Section 2.4 and Section 3.3 now report augmentation ablations and test-time robustness sensitivity. Minimal augmentation improved counting substantially (YOLO11n MAE 4.27 +- 0.60; YOLOv8n MAE 4.78 +- 0.69), while stronger augmentation did not improve the current test distribution. Robustness was also evaluated under brightness, contrast, Gaussian blur, Gaussian noise, and synthetic occlusion perturbations. We still acknowledge that these are controlled sensitivity checks, not external farm validation.

Comment 4: Computational efficiency is insufficiently analyzed.

Response: Section 2.5 and Table 5 now add parameter count, GFLOPs, fine-tuned checkpoint size, and recorded 20-epoch training time. YOLO11n has 2.59M parameters and 6.4 GFLOPs, while YOLOv8n has 3.01M parameters and 8.2 GFLOPs. We also softened deployment claims because CPU latency, edge-device FPS, memory use, and quantized inference were not measured.

Comment 5: Post-processing was not explored beyond confidence threshold selection.

Response: Section 2.4 and Section 3.3 now add a validation-selected confidence/IoU/NMS sweep. This reduced YOLOv8n MAE from 10.08 +- 1.55 to 4.21 +- 0.98 and changed the detector count ranking. We still identify soft-NMS, weighted box fusion, density-dependent thresholds, and segmentation-aware duplicate removal as future analyses.

Comment 6: Statistical analysis is needed.

Response: Section 3.4 and Table 5 now include post-hoc statistical comparisons. At the original default-training seed level, the YOLO11n-versus-YOLOv8n paired t-test was not significant because only three seeds were available (p = 0.211). After averaging absolute error by test base identity, YOLO11n significantly reduced count error compared with YOLOv8n (paired t-test p = 0.003, Cohen's dz = 0.61; sign test p = 0.002). For the reviewer experiments, minimal augmentation improved YOLOv8n MAE over default training (paired t-test p = 0.009), and the post-processing sweep favored YOLOv8n over YOLO11n by MAE (paired t-test p = 0.025). Wilcoxon/sign-test results are reported cautiously because only three paired seeds were available.

## Remaining Limitations Acknowledged in the Revised Manuscript

The revised manuscript explicitly acknowledges that the dataset is moderate in size, the annotation quality audit is limited, the detector comparison does not cover segmentation or density-map methods, robustness testing is limited to controlled perturbations rather than external farm data, post-processing is not fully optimized beyond confidence/IoU/NMS sweeps, and edge-device inference speed was not measured. These limitations are now presented as future-work priorities rather than being hidden or overclaimed.
