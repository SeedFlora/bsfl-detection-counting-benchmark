# Experiment Plan and Completion Status Before Submission

Status update: the reviewer-requested experiment batch has been run on 2026-06-01. Large YOLO training artifacts were stored outside the repository at `C:\BSF_reviewer_experiments`, while compact CSV/JSON summaries were copied back into `reports/reviewer_experiments/`.

## High Priority

1. Ulang training `YOLOv8n` dan `YOLO11n` minimal `3` seed.
   Tujuan:
   - menunjukkan stabilitas
   - melaporkan mean dan standard deviation

2. Tambahkan `YOLO11m` jika waktu dan VRAM cukup.
   Tujuan:
   - menguji apakah model yang sedikit lebih besar memberi gain yang konsisten

3. Lakukan `density-aware error analysis`.
   Caranya:
   - bagi image test menjadi `low`, `medium`, `high` density berdasarkan jumlah objek
   - hitung MAE per kelompok

4. Buat figure contoh prediksi.
   Perlu:
   - 3 contoh hasil bagus
   - 3 contoh hasil gagal atau over-count

5. Tambahkan penjelasan reproducibility.
   Cantumkan:
   - Docker workflow
   - GPU
   - package utama
   - split dataset

## Medium Priority

1. Jalankan evaluasi confidence sweep yang lebih rapat untuk counting dari detector. **Done** via confidence/IoU/NMS sweep.
2. Tambahkan satu baseline detector lain jika memungkinkan.
3. Uji augmentasi atau epoch lebih panjang pada `YOLO11n`. **Done** for YOLO11n and YOLOv8n across three seeds.

## Reviewer-Requested Optional Experiments

File eksperimen sudah disiapkan dan sudah dijalankan:

| Reviewer issue | Script | Perlu training? |
| --- | --- | --- |
| Confidence/NMS/post-processing sweep | `scripts/benchmark_yolo_postprocessing_sweep.py` | Tidak |
| Robustness brightness/contrast/blur/noise/occlusion | `scripts/benchmark_yolo_robustness.py` | Tidak |
| 50-epoch check dan augmentation ablation | `scripts/benchmark_yolo_reviewer_training.py` | Ya |
| One-command Docker launcher | `scripts/run_reviewer_experiments_gpu.ps1` | Opsional, hanya jika pakai `-RunTraining` |

Output utama:

- Post-processing: `reports/reviewer_experiments/postprocessing/reviewer_full_20260601_postprocessing/`
- Robustness: `reports/reviewer_experiments/robustness/reviewer_full_20260601_robustness/`
- Training ablation summary: `reports/reviewer_experiments/training/reviewer_full_20260601_training_c/`
- Statistical summary: `reports/paper_q4_prep/reviewer_experiment_results_summary.md`
- Statistical tests CSV: `reports/paper_q4_prep/statistical_tests.csv`
- Panduan lengkap: `reports/paper_q4_prep/reviewer_experiment_runbook.md`

## Low Priority

1. Bawa regresi ukuran ke appendix saja.
2. Tambahkan analisis inference speed bila target jurnal suka aspek deployment.

## Submission Readiness Check

Sebelum submit, idealnya naskah sudah punya:

- tabel hasil utama
- figure benchmark
- minimal satu analisis error
- minimal dua model deep learning yang diuji multi-seed
- diskusi limitasi data yang jujur
