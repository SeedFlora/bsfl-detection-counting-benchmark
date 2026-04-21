# Experiment Plan Before Submission

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

1. Jalankan evaluasi confidence sweep yang lebih rapat untuk counting dari detector.
2. Tambahkan satu baseline detector lain jika memungkinkan.
3. Uji augmentasi atau epoch lebih panjang pada `YOLO11n`.

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
