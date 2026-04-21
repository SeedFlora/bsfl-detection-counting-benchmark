# Figure Insert Guide

Use the split IEEE-ready figures in:

`D:/13359376/reports/paper_q4_prep/ieee_figures_split`

This folder is now the recommended figure source. Each chart is single-panel, has extra whitespace, and avoids subfigure labels.

## Recommended Main Figures

Use these first if the page limit is tight:

- [fig01_methodology_flowchart.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig01_methodology_flowchart.pdf)
- [fig02_detector_counting_pipeline.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig02_detector_counting_pipeline.pdf)
- [fig05_dataset_count_distribution.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig05_dataset_count_distribution.pdf)
- [fig06_method_family_mae.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig06_method_family_mae.pdf)
- [fig07_method_family_r2.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig07_method_family_r2.pdf)
- [fig13_yolo_count_mae.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig13_yolo_count_mae.pdf)
- [fig15_density_mae.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig15_density_mae.pdf)
- [fig18_scatter_yolo11n_seed2.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig18_scatter_yolo11n_seed2.pdf)
- [fig21_qualitative_success_failure_examples.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig21_qualitative_success_failure_examples.pdf)

## Dataset Figures

- [fig03_dataset_image_split.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig03_dataset_image_split.pdf)
- [fig04_dataset_instance_split.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig04_dataset_instance_split.pdf)
- [fig05_dataset_count_distribution.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig05_dataset_count_distribution.pdf)

## Classical Counting Figures

- [fig08_classical_baseline_mae.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig08_classical_baseline_mae.pdf)
- [fig09_top_classical_mae.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig09_top_classical_mae.pdf)
- [fig10_top_classical_r2.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig10_top_classical_r2.pdf)

## YOLO Multi-Seed Figures

- [fig11_yolo_map50.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig11_yolo_map50.pdf)
- [fig12_yolo_map50_95.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig12_yolo_map50_95.pdf)
- [fig13_yolo_count_mae.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig13_yolo_count_mae.pdf)
- [fig14_yolo_count_r2.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig14_yolo_count_r2.pdf)

## Error Analysis Figures

- [fig15_density_mae.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig15_density_mae.pdf)
- [fig16_density_exact_match.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig16_density_exact_match.pdf)
- [fig17_scatter_extra_trees.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig17_scatter_extra_trees.pdf)
- [fig18_scatter_yolo11n_seed2.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig18_scatter_yolo11n_seed2.pdf)

## Training And Qualitative Figures

- [fig19_training_map50.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig19_training_map50.pdf)
- [fig20_training_map50_95.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig20_training_map50_95.pdf)
- [fig21_qualitative_success_failure_examples.pdf](D:/13359376/reports/paper_q4_prep/ieee_figures_split/fig21_qualitative_success_failure_examples.pdf)

## Captions And LaTeX

- [figure_captions_ieee_split.md](D:/13359376/reports/paper_q4_prep/ieee_figures_split/figure_captions_ieee_split.md)
- [ieee_split_figure_snippets.tex](D:/13359376/reports/paper_q4_prep/ieee_figures_split/ieee_split_figure_snippets.tex)

## Regeneration

Run this command after updating benchmark outputs:

```powershell
python scripts\build_ieee_split_figures.py
```
