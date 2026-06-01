# IEEE Split Figure Captions

These captions match the single-panel figures in `ieee_figures_split`. The files avoid subfigure labels and provide extra whitespace to reduce clipping in IEEE layouts.

1. **Fig. 1. Experimental workflow.** Raw BSFL images and YOLO-format object annotations were curated into a single-class benchmark manifest, split by base image identity to reduce leakage, and evaluated using direct image-processing methods, feature-based count regressors, and detector-based counting.

2. **Fig. 2. Detector-based counting protocol.** Each retained detector output is interpreted as one larva after confidence filtering and standard YOLO post-processing; the resulting image-level count is compared with the annotation-derived ground-truth count.

3. **Fig. 3. Image split.** The benchmark contains 224 training, 52 validation, and 45 test images.

4. **Fig. 4. Annotated larvae per split.** The training, validation, and test splits contain 4757, 721, and 1134 annotated larvae, respectively.

5. **Fig. 5. Object-count distribution.** Larvae counts vary substantially across images, with high-density images forming a smaller but important subset that dominates severe counting errors.

6. **Fig. 6. Counting error by method family.** The best single YOLO11n run achieved the lowest MAE, while feature-based regressors remained strong non-deep-learning baselines.

7. **Fig. 7. Count-fit quality by method family.** HistGradientBoosting and the best YOLO11n run achieved the highest R2 values.

8. **Fig. 8. Classical counting baseline MAE.** Feature-based regressors outperformed direct thresholding, connected-component, blob, and watershed baselines.

9. **Fig. 9. Top classical regressor MAE.** Extra Trees achieved the lowest MAE among the classical count regressors.

10. **Fig. 10. Top classical regressor R2.** HistGradientBoosting achieved the highest R2 among the classical count regressors.

11. **Fig. 11. YOLO average detection mAP50.** Bars show mean and standard deviation across three seeds; YOLOv8n achieved the highest average mAP50 despite not producing the best average count accuracy.

12. **Fig. 12. YOLO average detection mAP50-95.** Bars show mean and standard deviation across three seeds; YOLOv8n retained the strongest stricter localization score.

13. **Fig. 13. YOLO average counting MAE.** Bars show mean and standard deviation across three seeds; YOLO11n achieved lower average detector-based counting error than YOLOv8n.

14. **Fig. 14. YOLO average counting R2.** Bars show mean and standard deviation across three seeds; YOLO11n achieved higher average count-fit quality even though YOLOv8n had stronger mAP.

15. **Fig. 15. Counting error by density.** Counting error increased strongly in high-density scenes, indicating that overlap and duplicate detections are the main remaining failure modes.

16. **Fig. 16. Exact-count agreement by density.** Exact-count agreement decreased in high-density scenes, showing that crowded trays require additional post-processing or alternative counting models.

17. **Fig. 17. Extra Trees count prediction.** Predicted and ground-truth counts are compared on the test set for the strongest classical MAE baseline.

18. **Fig. 18. YOLO11n seed 2 count prediction.** Predicted and ground-truth counts are compared on the test set for the best detector-based single run.

19. **Fig. 19. YOLO validation mAP50 convergence.** Validation mAP50 curves show that all detector runs improved within the 20-epoch training budget, with the best validation mAP50 occurring at the final epoch for the multi-seed runs.

20. **Fig. 20. YOLO validation mAP50-95 convergence.** Validation mAP50-95 curves show stricter localization convergence across seeds and support treating the 20-epoch setup as a controlled lightweight benchmark rather than a fully exhausted training schedule.

21. **Fig. 21. Qualitative success and failure examples.** Representative test examples illustrate exact-count successes in less ambiguous scenes and crowded-scene over-counting failures caused by overlapping larval bodies.
