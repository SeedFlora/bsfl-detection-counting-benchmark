# IEEE Figure Captions

Use the `.pdf` files for IEEE whenever possible because they preserve vector text and lines. Use the `.png` files as 600 dpi raster fallbacks.

1. **Fig. 1. Overall experimental workflow.** Raw BSF larvae images and YOLO-format annotations were curated into a benchmark-ready manifest, split into train, validation, and test partitions, and evaluated using classical image-processing baselines, feature-based count regressors, and YOLO detector-based counting.

2. **Fig. 2. Detector-based counting protocol.** Each predicted bounding box is interpreted as one larva. The resulting predicted count is compared with the number of annotated larvae per image using MAE, RMSE, R2, and exact-count agreement.

3. **Fig. 3. Dataset overview.** The benchmark contains 224 training, 52 validation, and 45 test images, corresponding to 4757, 721, and 1134 annotated larvae, respectively.

4. **Fig. 4. Best representative methods by family.** Direct classical computer vision was substantially weaker than feature-based count regressors and detector-based counting. The best single detector-based run (YOLO11n seed 2) achieved the lowest test MAE.

5. **Fig. 5. Classical counting benchmark.** Feature-based regressors were the strongest non-deep-learning methods. Extra Trees produced the lowest classical MAE, while HistGradientBoosting produced the strongest classical RMSE/R2 behavior.

6. **Fig. 6. Multi-seed YOLO evaluation.** Bars show mean +/- standard deviation across three random seeds. YOLOv8n achieved the strongest average detection scores, while YOLO11n achieved the strongest average detector-based counting error.

7. **Fig. 7. Density-aware counting analysis.** Counting error increased substantially in high-density scenes, indicating that crowded larvae regions are the main remaining failure mode.

8. **Fig. 8. Predicted versus ground-truth counts.** Scatter plots compare the strongest classical MAE baseline (Extra Trees) against the best single detector-based counting run (YOLO11n seed 2) on the test set.

9. **Fig. 9. YOLO validation convergence across seeds.** Validation mAP curves show that both lightweight detectors converged within the 20-epoch training budget, with shaded regions representing one standard deviation across seeds.

10. **Fig. 10. Qualitative success and failure examples.** Representative test images illustrate exact or near-exact counting cases and failure cases, especially over-counting in crowded scenes.
