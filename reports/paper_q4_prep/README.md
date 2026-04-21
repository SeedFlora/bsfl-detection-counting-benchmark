# Q4 Paper Prep

Folder ini merangkum bahan yang paling siap dipakai untuk menyusun paper berbasis task `larvae detection and counting`.

## Posisi Naskah yang Disarankan

Fokus paper sebaiknya:

- `automatic larvae detection and counting`
- studi komparatif `classical methods vs lightweight deep detectors`
- kontribusi utama pada `dataset curation`, `benchmark reproducible`, dan `error-aware evaluation`

Jangan jadikan `regression` sebagai tema utama, karena hasilnya masih lebih lemah dan belum sekuat jalur counting.

## Bukti Utama Saat Ini

- Dataset deteksi berisi `321` gambar dengan `6,612` anotasi objek.
- Split evaluasi saat ini: `224` train, `52` validation, `45` test.
- Metode counting klasik dengan `MAE` terbaik saat ini adalah `extra_trees_count_regression` dengan `MAE 7.03` dan `R2 0.823`.
- Metode klasik dengan `RMSE` terendah dan `R2` tertinggi saat ini adalah `hist_gradient_boosting_count_regression` dengan `RMSE 14.37` dan `R2 0.859`.
- Untuk `average detection quality`, `YOLOv8n` tetap sangat kuat dengan `mAP50 0.864`, `mAP50-95 0.702`, `MAE 8.58`, dan `R2 0.772` pada single run terbaik yang sudah terdokumentasi.
- Untuk `detector-based counting`, `YOLO11n` sekarang menjadi kandidat terkuat setelah evaluasi multi-seed, dan best single run-nya mencapai `MAE 6.07` serta `R2 0.861`.

## Isi Folder

- `results_table_detection_counting.csv`
  Tabel hasil utama lintas metode.
- `tables_for_manuscript.md`
  Tabel siap-tempel untuk manuskrip.
- `tables_for_manuscript_multiseed.md`
  Tabel revisi setelah penguatan multi-seed.
- `abstract_draft.md`
  Draft abstrak bahasa Inggris.
- `paper_outline.md`
  Outline paper, usulan judul, dan struktur isi.
- `experiment_plan.md`
  Daftar eksperimen tambahan yang paling penting sebelum submit.
- `strengthening_summary.md`
  Ringkasan hasil eksperimen penguatan multi-seed, density analysis, dan qualitative figure.
- `key_results.json`
  Ringkasan metrik terbaik.

## Status

Materi ini sekarang sudah lebih kuat untuk paper Q4 bergaya `applied/comparative study`, terutama setelah eksperimen multi-seed. Meski begitu, naskah tetap akan lebih aman jika ditambah sitasi literatur, figure kualitatif yang dipoles, dan bila memungkinkan satu putaran validasi tambahan.
