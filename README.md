# PsyDefDetect-Trajectory-Calibration

Pipeline eksperimen deteksi level *Defense Mechanism* pada percakapan dukungan emosional berbasis pemodelan lintasan dialog (*dialogue trajectory modeling*) dan kalibrasi logit terstruktur (*structured calibration*).

Sesuai dengan spesifikasi [prd_pipeline_eksperimen_psydefdetect.md](prd_pipeline_eksperimen_psydefdetect.md).

---

## 1. Rumusan Masalah (Research Questions)

* **RQ1 (Pemodelan Lintasan Dialog Eksplisit):** Mengukur peningkatan *Macro-F1* (kelas 1–8 dan 0–8) serta penurunan kesalahan klasifikasi pada kelas yang saling tumpang tindih (**C5 *Neurotic***, **C6 *Obsessional***, dan **C7 *High-Adaptive***) dengan menambahkan model tahap kedua berbasis fitur lintasan (prediksi 1–3 ucapan *seeker* sebelumnya + posisi *turn* absolut) di atas *baseline* Qwen3.
* **RQ2 (Kalibrasi Logit Terstruktur):** Mengukur apakah kalibrasi logit berparameter ringkas (*structured calibration*) yang dievaluasi menggunakan *nested grouped cross-validation* mampu menyempitkan *generalization gap* antara skor validasi dan skor pengujian dibandingkan pencarian bias acak (*random bias search*).

---

## 2. Struktur Modul Notebook

Pipeline ini dirancang modular dalam 5 Jupyter Notebook:

1. **`01_eda_baseline.ipynb` (CPU Lokal):**
   * Validasi skema dataset (`train.json` & `test.json`).
   * Analisis irisan 189 `dialogue_id` antara train dan test.
   * Analisis lintasan empiris (matriks transisi 9×9 DMRS) dan audit panjang token Qwen.
   * Penetapan fold permanen: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` disimpan di `artifacts/splits/train_folds.parquet`.
   * *Sanity baseline* CPU: TF-IDF (20.000 n-gram) + L2 Logistic Regression.

2. **`02_stage1_qwen3_qlora.ipynb` (GPU Vast.ai $\ge 24\text{ GB}$ VRAM):**
   * *Fine-tuning* `Qwen/Qwen3-8B` (atau 4B) dengan QLoRA 4-bit NF4 (`r=128`, `alpha=256`).
   * *Weighted Cross-Entropy Loss* berbobot *inverse-square-root* dengan *label smoothing* $\epsilon = 0.05$.
   * *Paged AdamW 8-bit*, *cosine learning rate scheduler*, *gradient checkpointing*, *dynamic batch padding*.
   * Menyimpan logit *Out-of-Fold* (`oof_logits.npy`) dan inferensi test per fold (`test_logits_by_fold.npy`).

3. **`03_stage2_sequential_trajectory.ipynb` (CPU Lokal):**
   * Rekonstruksi urutan giliran dialog *seeker* (`seeker_turn_idx`, `abs_turn_idx`).
   * Rekayasa fitur lintasan murni OOF ($t-1, t-2, t-3$), *DMRS maturity score* $E_t$, perubahan $\Delta E_t$, dan sinyal ketidakpastian.
   * Pelatihan model sekuensial: LightGBM, CatBoost, dan Multinomial Logistic Regression.
   * *Held-out test routing* untuk evaluasi `test.json`.

4. **`04_stage3_structured_calibration.ipynb` (CPU Lokal):**
   * *Nested Grouped Cross-Validation* (5 *outer fold* $\times$ 4 *inner fold*).
   * Kalibrasi berparameter ringkas:
     * Method A: Prior Logit Adjustment (1 parameter $\tau$)
     * Method B: Temperature Scaling + Prior Adjustment (2 parameter $T, \tau$)
     * Method C: Structured C0/C7 Gate + Prior Adjustment (3 parameter $b_0, b_7, \tau$)
     * Method D: Multiclass Platt Scaling / L2 Logistic Regression
     * Baseline: Random Bias Search (22.000 percobaan)
   * Menyimpan parameter kalibrasi terbaik di `artifacts/results/best_calibration_params.json`.

5. **`05_ablation_final_eval.ipynb` (CPU Lokal):**
   * Unit test otomatis *Zero-Leakage* dan *Causal Integrity*.
   * Tabel ablasi 4 kondisi:
     1. Baseline (Qwen3-8B Argmax)
     2. Baseline + Structured Calibration (RQ2)
     3. Baseline + Sequential Stage (RQ1)
     4. Full Pipeline (RQ1 + RQ2)
   * Evaluasi satu kali pada *held-out test set* (dengan pengukuran inflasi split dialog).
   * *Bootstrap 95% Confidence Interval* (1.000 iterasi).
   * Analisis kesalahan mendalam pada C5/C6/C7 dan pemulihan kelas minoritas (C2, C5, C8).

---

## 3. Struktur Artefak

```text
artifacts/
├── splits/
│   └── train_folds.parquet          # id, dialogue_id, fold_id, seeker_turn_idx, abs_turn_idx
├── stage1_qwen3_8b/
│   └── seed_42/
│       ├── oof_logits.npy           # [1864, 9] (dihasilkan oleh Notebook 02)
│       └── test_logits_by_fold.npy  # [5, 472, 9] (dihasilkan oleh Notebook 02)
├── stage2_sequential/
│       ├── oof_logits_lgbm.npy      # [1864, 9] (dihasilkan oleh Notebook 03)
│       ├── oof_logits_catboost.npy  # [1864, 9] (dihasilkan oleh Notebook 03)
│       └── test_logits_routed_*.npy # [472, 9] (dihasilkan oleh Notebook 03)
└── results/
        ├── tfidf_lr_baseline.csv
        ├── label_distribution.png
        ├── transition_matrix.png
        ├── turn_length_per_class.png
        ├── token_length_distribution.png
        ├── best_calibration_params.json
        ├── calibration_nested_cv.csv
        ├── ablation_summary.csv
        └── per_class_metrics.csv
```

---

## 4. Cara Menjalankan

### A. Lingkungan Lokal (CPU)
Notebook 01 sudah selesai dijalankan dan menghasilkan `artifacts/splits/train_folds.parquet`.

### B. Lingkungan GPU Cloud (Vast.ai)
1. Salin repositori ini ke instance Vast.ai ($\ge 24\text{ GB}$ VRAM, mis. RTX 3090/4090).
2. Install dependensi:
   ```bash
   pip install torch transformers peft bitsandbytes accelerate pandas numpy scikit-learn tqdm
   ```
3. Jalankan `02_stage1_qwen3_qlora.ipynb` hingga selesai seluruh 5 fold.
4. Salin hasil `artifacts/stage1_qwen3_8b/seed_42/oof_logits.npy` dan `test_logits_by_fold.npy` kembali ke repositori lokal.

### C. Eksekusi Tahap Lanjutan (CPU Lokal)
Setelah file logit Tahap 1 tersedia:
1. Jalankan `03_stage2_sequential_trajectory.ipynb`.
2. Jalankan `04_stage3_structured_calibration.ipynb`.
3. Jalankan `05_ablation_final_eval.ipynb` untuk mendapatkan tabel dan gambar hasil akhir Bab 4 skripsi.
