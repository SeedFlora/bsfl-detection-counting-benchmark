# Paper Outline

## Recommended Framing

Posisikan naskah sebagai:

- `applied computer vision study`
- `reproducible benchmark for larvae detection and counting`
- fokus utama pada `counting through detection`

## Alternative Title Options

1. Automatic Detection and Counting of Black Soldier Fly Larvae Using Classical Computer Vision and Lightweight YOLO Models
2. A Reproducible Benchmark for Black Soldier Fly Larvae Counting with Classical and Deep Learning Methods
3. Comparative Evaluation of Classical Vision and YOLO-Based Models for Larvae Detection and Counting

## Claimed Contributions

1. Menyusun dataset deteksi dan counting yang terkurasi dari data proyek dengan split evaluasi yang konsisten.
2. Membandingkan baseline klasik dan model YOLO ringan dalam satu workflow yang reproducible berbasis Docker dan GPU.
3. Menunjukkan trade-off antara akurasi counting dan kualitas deteksi pada beberapa varian model ringan.

## Section-by-Section Structure

### 1. Introduction

- Latar belakang pentingnya monitoring larva otomatis.
- Keterbatasan counting manual.
- Alasan membandingkan metode klasik dan deep learning.
- Ringkasan kontribusi paper.

### 2. Materials and Methods

#### 2.1 Dataset Preparation

- Sumber data dari dataset proyek.
- Ringkasan kurasi dan remapping label.
- Statistik final:
  - `321` images
  - `6,612` annotated larvae
  - split `224/52/45` untuk train/val/test

#### 2.2 Detection and Counting Tasks

- Definisi task deteksi objek.
- Definisi task counting per image.
- Metrik:
  - Precision
  - Recall
  - mAP50
  - mAP50-95
  - MAE
  - RMSE
  - R2

#### 2.3 Compared Methods

- Otsu connected components
- Adaptive connected components
- Watershed counting
- Random forest count regression
- YOLOv8n
- YOLO11n
- YOLO11s

#### 2.4 Experimental Setup

- Training dan evaluasi via Docker GPU.
- Perangkat GPU lokal.
- Epoch, batch size, image size, dan confidence tuning untuk counting dari detector.

### 3. Results

#### 3.1 Dataset Summary

Gunakan angka dari:

- `reports/preparation_summary.json`

#### 3.2 Main Benchmark Results

Gunakan:

- `reports/paper_q4_prep/results_table_detection_counting.csv`
- `reports/paper_q4_prep/tables_for_manuscript.md`

Poin naratif yang perlu ditegaskan:

- Random forest terbaik untuk counting murni.
- YOLOv8n terbaik untuk skenario end-to-end deteksi + counting.
- YOLO11n kompetitif.
- YOLO11s tidak otomatis lebih baik untuk counting walau localization sedikit naik.

#### 3.3 Visual Evaluation

Gunakan figure dari:

- `reports/yolo_benchmark/yolo11_gpu_benchmark/plots/detection_metrics_comparison.png`
- `reports/yolo_benchmark/yolo11_gpu_benchmark/plots/counting_metrics_comparison.png`
- `reports/yolo_benchmark/yolo11_gpu_benchmark/plots/training_curve_map50.png`
- `reports/yolo_benchmark/yolo11_gpu_benchmark/plots/yolo11n_test_count_scatter.png`
- `reports/yolo_benchmark/yolo11_gpu_benchmark/plots/yolo11s_test_count_scatter.png`

### 4. Discussion

- Mengapa handcrafted regression masih unggul di counting murni.
- Mengapa model deteksi tetap lebih menarik untuk aplikasi nyata.
- Error cenderung meningkat pada image dengan objek padat.
- Keterbatasan dataset dan ukuran test set.

### 5. Conclusion

- Counting otomatis feasible.
- YOLO ringan sudah cukup kuat untuk use case praktis.
- Perlu evaluasi multi-seed dan data yang lebih besar untuk generalisasi yang lebih kuat.

## Minimum Figure/Table Set

### Tables

1. Dataset summary
2. Main benchmark results
3. Optional ablation or additional experiment table

### Figures

1. Detection metric comparison
2. Counting metric comparison
3. Training curve
4. Good prediction examples
5. Failure case examples
