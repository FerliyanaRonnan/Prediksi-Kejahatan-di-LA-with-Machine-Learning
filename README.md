# LAPD Crime Data Prediksi Penangkapan Pelaku Kejahatan

Submission untuk babak final **Data Science Competition Crime Analytics 2024**. Notebook ini menyajikan pipeline lengkap untuk memprediksi apakah suatu kejahatan akan berakhir dengan penangkapan pelaku (`is_arrested`), dari dataset kejahatan Los Angeles (LAPD) periode Januari 2020 – September 2024 yang terdiri dari 982.638 laporan kejahatan, menggunakan pendekatan binary classification berbasis gradient boosting.

## Gambaran Masalah

| Parameter | Detail |
|---|---|
| Data historis | 982.638 laporan kejahatan, Januari 2020 – September 2024 |
| Target | `is_arrested` (biner): 1 = ditangkap (Adult/Juv Arrest), 0 = tidak ditangkap |
| Class imbalance | 9,0% kasus ditangkap (88.351) vs 91,0% kasus tidak ditangkap (894.287) |
| Metrik evaluasi | F1 Weighted (primer), F1 Macro, ROC-AUC |
| Tantangan utama | Class imbalance berat; threshold default 0,5 tidak optimal untuk data imbalanced |

## Pendekatan

Digunakan ensemble dua model komplementer dengan threshold optimization.

| Model | Peran |
|---|---|
| LightGBM | Model utama, `class_weight='balanced'` + Optuna hyperparameter tuning (50 trials) |
| XGBoost | Model pembanding, `scale_pos_weight`, diversifikasi ensemble |
| Dummy Classifier | Baseline minimum (selalu memprediksi kelas mayoritas) |

Kedua model dikombinasikan sebagai **soft voting ensemble** (LightGBM 60% + XGBoost 40%). Threshold keputusan dioptimasi (bukan default 0,5) dengan mencari nilai yang memaksimalkan F1 Macro pada validation set ditemukan threshold optimal 0,73.

## Struktur Notebook

| Cell | Deskripsi |
|---|---|
| 0 | Instalasi library dan setup konfigurasi global |
| 1 | Load data dan preprocessing |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Feature engineering dan label encoding |
| 4 | Train / Validation / Test split |
| 5 | Baseline model (Dummy Classifier) |
| 6 | Training LightGBM awal (sebelum tuning) |
| 7 | Training XGBoost model pembanding |
| 8 | Perbandingan model sebelum hyperparameter tuning |
| 9 | Optuna hyperparameter tuning (50 trials) |
| 10 | Training LightGBM final dengan parameter terbaik |
| 11 | Threshold optimization |
| 12 | Feature importance & SHAP analysis |
| 13 | Ensemble LightGBM + XGBoost (soft voting) |
| 14 | Evaluasi final dan perbandingan semua model |
| 15 | Visualisasi final: confusion matrix & ROC curve |
| 16 | Actionable insights & rekomendasi kebijakan |
| 17 | Simpan model ke Google Drive dan artefak |

## Data

### Sumber

- **LAPD Crime Data** dari Los Angeles Open Data Portal 982.638 baris × 28 kolom mentah, dengan atribut:
  - **Identitas kejadian**: nomor laporan, tanggal lapor, tanggal & waktu kejadian
  - **Lokasi**: kode area (21 divisi LAPD), koordinat LAT/LON, kode premis
  - **Kejahatan**: kode & deskripsi kejahatan (140 jenis), kode senjata
  - **Korban**: umur, jenis kelamin, ras/etnis
  - **Status**: status penanganan kasus — basis pembentukan label `is_arrested`

### Preprocessing

1. Buang 8 kolom redundan/missing masif: `DR_NO`, `Rpt Dist No`, `Crm Cd 2`–`4`, `Cross Street`, `Mocodes`.
2. Parsing `DATE OCC`/`TIME OCC` → fitur temporal (`occ_year`, `occ_month`, `occ_hour`, `occ_dayofweek`, dst).
3. Bangun label `is_arrested`: 1 jika status `Adult Arrest`/`Juv Arrest`, 0 untuk status lainnya.
4. Koordinat (0,0) → `NaN` (invalid LAPD); hitung `dist_from_center_km` dan clustering `geo_cluster` (K-Means, 30 zona).
5. Tangani missing value senjata/korban; sederhanakan 140 jenis kejahatan → 9 kategori (`crime_category`), premis → 8 kategori (`premis_category`).

**Mengapa tidak ada normalisasi (StandardScaler/MinMaxScaler)?** Model utama (LightGBM, XGBoost) berbasis pohon keputusan yang *scale-invariant* pemisahan data dilakukan berdasarkan nilai ambang (threshold), bukan jarak antar titik, sehingga normalisasi tidak memberikan manfaat tambahan terhadap performa model.

### Feature Engineering

27 fitur final dibagi 6 kelompok:

| Kelompok | Jumlah | Contoh Fitur |
|---|---|---|
| Temporal | 12 | `occ_hour`, `season`, `is_weekend`, `is_night`, `report_delay_days` |
| Lokasi | 3 | `geo_cluster`, `dist_from_center_km`, `AREA` |
| Kejahatan | 3 | `crime_category`, `Crm Cd`, `Part 1-2` |
| Premis | 2 | `premis_category`, `Premis Cd` |
| Senjata | 2 | `has_weapon`, `Weapon Used Cd` |
| Korban | 3 | `Vict Age`, `Vict Sex`, `Vict Descent` |

## Hasil

### Performa Model (Test Set)

| Model | F1 Weighted | F1 Macro | ROC-AUC |
|---|---|---|---|
| Dummy Baseline | 0,8673 | 0,4806 | 0,5000 |
| LightGBM (awal) | 0,7882 | 0,5882 | 0,8300 |
| XGBoost | 0,7853 | 0,5900 | 0,8286 |
| LightGBM (tuned) | 0,7995 | 0,5983 | 0,8324 |
| **LightGBM (tuned + threshold 0,73)** | **0,8801** | **0,6540** | **0,8324** |
| Ensemble (LightGBM 60% + XGBoost 40%) | 0,8797 | 0,6531 | 0,8320 |

**Model terbaik: LightGBM (tuned + threshold 0,73)** — dipilih berdasarkan F1 Macro tertinggi.

### Classification Report — Model Terbaik

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Tidak Ditangkap | 0,94 | 0,92 | 0,93 | 134.144 |
| Ditangkap | 0,34 | 0,43 | 0,38 | 13.252 |

Recall kelas "Ditangkap" naik dari **0% (baseline)** menjadi **43%** — peningkatan signifikan dalam mendeteksi kasus minoritas, dengan trade-off precision yang relatif rendah (34%, banyak false positive).

### Hyperparameter Terbaik (Optuna, 50 trials)

ROC-AUC validasi terbaik: **0,831**. Parameter kunci: `num_leaves=106`, `max_depth=10`, `learning_rate≈0,06`, `feature_fraction=0,656`, `bagging_fraction=0,728`.

## Temuan Utama (EDA & Feature Importance)

**Top 5 fitur paling berpengaruh** (LightGBM Tuned): `dist_from_center_km`, `Crm Cd`, `Vict Age`, `Premis Cd`, `occ_day`.

**Faktor risiko dominan:**
- Arrest rate sangat bervariasi per jenis kejahatan — Pembunuhan tertinggi (~60%), Pencurian/Perampokan terendah (~5%) meski mendominasi 57% dari seluruh kasus (*opportunity gap* terbesar).
- Keberadaan senjata melipatgandakan arrest rate: 17,1% (ada senjata) vs 4,9% (tanpa senjata) bukti fisik balistik memudahkan penyelidikan.
- Variasi arrest rate antar wilayah: Harbor, Mission, dan N Hollywood punya arrest rate tertinggi (~12%).
- Berdasarkan interpretasi SHAP (TreeExplainer, sampel 3.000 data test): jenis kejahatan, jarak dari pusat kota, dan keberadaan senjata paling konsisten mendorong prediksi ke arah "ditangkap".

**Catatan kehati-hatian:** kategori `NARKOBA` hanya memiliki 12 kasus di dataset — klaim arrest rate untuk kategori ini secara statistik kurang reliable karena sampel terlalu kecil.

## Model yang Diekspor

Semua artefak tersimpan di Google Drive (`/Crime_Data/`):

| File | Format | Isi |
|---|---|---|
| `lgbm_tuned.pkl` | Pickle | LightGBM model final (tuned) |
| `xgb_model.pkl` | Pickle | XGBoost model pembanding |
| `label_encoders.pkl` | Pickle | Dictionary `LabelEncoder` per kolom kategorikal |
| `best_threshold.pkl` | Pickle | Threshold optimal (0,73) untuk inference |
| `model_comparison.csv` | CSV | Tabel perbandingan semua model |

## Instalasi

```bash
pip install lightgbm xgboost optuna shap
```

Library lain yang digunakan: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

## Cara Menjalankan

Jalankan notebook secara berurutan dari Cell 0 hingga Cell 17. Notebook dikembangkan di Google Colab dan memuat data dari Google Drive — sesuaikan `DATA_PATH` jika dijalankan di environment lain.

```
.
├── notebook.ipynb
├── archive.zip            # LAPD Crime Data mentah
└── models/                # dibuat otomatis saat Cell 17 dijalankan
    ├── lgbm_tuned.pkl
    ├── xgb_model.pkl
    ├── label_encoders.pkl
    ├── best_threshold.pkl
    └── model_comparison.csv
```

### Inference pada Data Baru

```python
import joblib

lgbm_model = joblib.load('models/lgbm_tuned.pkl')
le_dict = joblib.load('models/label_encoders.pkl')
threshold_data = joblib.load('models/best_threshold.pkl')
best_thr = threshold_data['threshold']

y_prob = lgbm_model.predict_proba(X_new)[:, 1]
y_pred = (y_prob >= best_thr).astype(int)
```

## Catatan Metodologis

- **Random split, bukan temporal split**: data dibagi 70/15/15 dengan stratified random split untuk menjaga proporsi kelas, bukan dipisah berdasarkan tahun. Cocok untuk evaluasi klasifikasi umum, namun untuk simulasi forecasting kasus di masa depan, temporal split (train 2020–2023, test 2024) akan lebih representatif.
- **Tidak ada algoritma bagging (Random Forest) yang diuji** sebagai pembanding — fokus pada keluarga boosting karena secara struktural lebih mengurangi bias terhadap kelas minoritas dibanding bagging.
- **Bobot ensemble (60:40) dipilih heuristik**, bukan hasil grid search sistematis seperti threshold optimization. Karena gap performa LightGBM vs XGBoost relatif kecil (ROC-AUC 0,832 vs 0,829), sensitivitas hasil akhir terhadap bobot diperkirakan rendah.
- **Precision rendah (34%)** pada kelas "Ditangkap" adalah trade-off yang disengaja demi recall lebih tinggi — model ini ditujukan sebagai alat bantu prioritisasi investigasi, bukan dasar tunggal keputusan terhadap individu.
- Label `is_arrested` mencerminkan keputusan kepolisian historis yang berpotensi mengandung bias — model sebaiknya tidak digunakan untuk profiling individu tanpa pengawasan etis tambahan.
