# BSFMaggot

Workspace untuk computer vision pada larva Black Soldier Fly (BSF/Maggot) — detection, counting, regression, dan size classification.

## Dataset

Dataset **BSF_Larvae_v1** tersedia di Zenodo:
**[https://zenodo.org/records/13359376](https://zenodo.org/records/13359376)**

Download dan ekstrak ke `data/raw/` sebelum menjalankan pipeline.

---

Workspace ini menyiapkan tiga task utama dari dataset larva:

- `detection/counting` dari `BSF_Larvae_v1`
- `regression` untuk `weight` dan `length`
- `size classification` (`small`, `medium`, `large`)

Ada juga baseline eksplorasi untuk `sex classification`, tetapi itu bukan task utama karena sinyal visualnya terlihat lebih lemah.

## Struktur Data

- `data/raw/`
  - hasil ekstraksi zip mentah
- `data/processed/`
  - manifest CSV, cache fitur, dan dataset deteksi yang sudah siap pakai
- `reports/`
  - metrik baseline dan ringkasan audit
- `models/`
  - model baseline yang tersimpan

## Quick Start

Install dependensi:

```bash
pip install -r requirements.txt
pip install -e .
```

Siapkan semua manifest:

```bash
python scripts/prepare_datasets.py
```

Jalankan semua baseline klasik:

```bash
python scripts/run_all_baselines.py
```

Atau jalankan per task:

```bash
python scripts/train_regression_baseline.py
python scripts/train_size_classification_baseline.py
python scripts/train_sex_classification_baseline.py
```

Benchmark 5 metode untuk counting dan 5 metode untuk regression/size classification:

```bash
python scripts/benchmark_detection_counting_methods.py
python scripts/benchmark_regression_size_methods.py
```

## Detection / Counting

Dataset deteksi dipersiapkan ke:

- `data/processed/detection_counting/images/{train,val,test}`
- `data/processed/detection_counting/labels/{train,val,test}`
- `data/processed/detection_counting/dataset.yaml`

Karena label asli `BSF_Larvae_v1` memiliki class id yang tidak konsisten untuk sebagian image dasar, pipeline ini meremap semua instance menjadi satu kelas `larva`. Itu membuat dataset lebih stabil untuk task `detection/counting`.

Training YOLO sekarang sudah termasuk dalam dependensi utama. Setelah `pip install -r requirements.txt`, script ini bisa langsung dipakai:

```bash
python scripts/train_detection_yolo.py --epochs 20
```

Untuk membandingkan beberapa varian YOLO sekaligus dan membuat grafik evaluasi:

```bash
python scripts/benchmark_yolo_variants.py --models yolo11n.pt yolo11s.pt --epochs 20 --train-device 0 --eval-device cuda --workers 2 --batch 16
```

Output benchmark akan disimpan ke:

- `reports/yolo_benchmark/<run_name>/comparison.csv`
- `reports/yolo_benchmark/<run_name>/summary.json`
- `reports/yolo_benchmark/<run_name>/plots/*.png`

## Docker

Container baseline CPU:

```bash
docker build -t larvae-cv .
docker run --rm -it -v ${PWD}:/workspace larvae-cv
```

Container GPU untuk YOLO dan regresi:

```bash
docker build -f Dockerfile.gpu -t larvae-cv-gpu .
docker run --rm --gpus all --shm-size=8g -v ${PWD}:/workspace -w /workspace larvae-cv-gpu python scripts/benchmark_yolo_variants.py --models yolo11n.pt yolo11s.pt --epochs 20 --train-device 0 --eval-device cuda --workers 2 --batch 16
```
