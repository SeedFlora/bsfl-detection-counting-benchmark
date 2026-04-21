# Strengthening Summary

Eksperimen penguatan dilakukan dengan:

- `YOLOv8n` dan `YOLO11n`
- `3` seed per model: `0`, `1`, `2`
- `20` epoch per run
- Docker GPU workflow yang sama seperti eksperimen utama

Artefak lengkap ada di:

- [summary.json](D:/13359376/reports/yolo_multiseed/yolo_multiseed_strengthening/summary.json)
- [run_level_results.csv](D:/13359376/reports/yolo_multiseed/yolo_multiseed_strengthening/run_level_results.csv)
- [aggregate_results.csv](D:/13359376/reports/yolo_multiseed/yolo_multiseed_strengthening/aggregate_results.csv)
- [density_summary.csv](D:/13359376/reports/yolo_multiseed/yolo_multiseed_strengthening/density_summary.csv)
- [plots](D:/13359376/reports/yolo_multiseed/yolo_multiseed_strengthening/plots)

## Main Takeaways

1. `YOLO11n` menjadi detector-counter terbaik secara `counting average`.
   - test count `MAE mean = 7.99 +- 1.71`
   - test count `R2 mean = 0.733 +- 0.128`

2. `YOLOv8n` tetap menjadi model terbaik secara `average detection quality`.
   - `mAP50 mean = 0.860 +- 0.010`
   - `mAP50-95 mean = 0.697 +- 0.010`

3. `Best single deep-learning run` sekarang adalah `YOLO11n seed 2`.
   - `MAE = 6.07`
   - `RMSE = 14.27`
   - `R2 = 0.861`
   - hasil ini lebih baik daripada semua baseline counting klasik yang diuji

5. Baseline counting klasik terbaru sekarang lebih kuat daripada versi awal.
   - `Extra Trees` memberi `MAE` klasik terbaik: `7.03`
   - `HistGradientBoosting` memberi `RMSE` terendah dan `R2` klasik tertinggi: `14.37` dan `0.859`

4. Multi-seed evaluation menunjukkan hasil deep learning memang `promising`, tetapi masih punya `variance`.
   - ini justru bagus untuk paper karena sekarang kita bisa melaporkan stabilitas, bukan hanya single lucky run

## Density Analysis

Secara `MAE`, kedua model memburuk saat jumlah objek meningkat, tetapi `YOLO11n` konsisten lebih baik pada semua density bin:

- low density:
  - `YOLO11n MAE mean = 2.38`
  - `YOLOv8n MAE mean = 2.60`
- medium density:
  - `YOLO11n MAE mean = 5.04`
  - `YOLOv8n MAE mean = 6.38`
- high density:
  - `YOLO11n MAE mean = 16.56`
  - `YOLOv8n MAE mean = 21.27`

Interpretasi penting:

- error counting meningkat tajam pada image padat
- ini cocok dijadikan salah satu poin diskusi utama di paper

## Qualitative Figure

Figure otomatis contoh prediksi bagus dan gagal sudah dibuat di:

- [qualitative_success_failure_montage.png](D:/13359376/reports/yolo_multiseed/yolo_multiseed_strengthening/plots/qualitative_success_failure_montage.png)

Daftar image yang dipakai ada di:

- [qualitative_examples.csv](D:/13359376/reports/yolo_multiseed/yolo_multiseed_strengthening/qualitative_examples.csv)

## Updated Position for the Paper

Setelah multi-seed:

- paper jadi `lebih kuat` untuk level Q4 applied study
- narasi yang paling aman sekarang:
  - `YOLO11n` terbaik untuk counting rata-rata
  - `YOLOv8n` terbaik untuk metrik deteksi rata-rata
  - high-density scenes tetap menjadi failure mode utama
