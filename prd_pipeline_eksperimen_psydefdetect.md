# Product Requirements Document (PRD)

**Nama Proyek:** Pipeline Eksperimen Deteksi Level *Defense Mechanism* pada Percakapan Dukungan Emosional (`PsyDefDetect-Trajectory-Calibration`)  
**Target Artefak:** Modular Jupyter Notebook (`01_eda_baseline.ipynb`, `02_stage1_qwen3_qlora.ipynb`, `03_stage2_sequential_trajectory.ipynb`, `04_stage3_structured_calibration.ipynb`, `05_ablation_final_eval.ipynb`) atau satu *end-to-end notebook* terstruktur dengan *checkpointing*.  
**Target Lingkungan Eksekusi:** CPU lokal (untuk EDA, validasi skema, fitur tabular, dan kalibrasi) + GPU Cloud Vast.ai ($\ge 24\text{ GB}$ VRAM, mis. RTX 3090/4090, untuk *fine-tuning* Qwen3-4B dan Qwen3-8B).

---

## 1. Tujuan & Ruang Lingkup Penelitian

Membangun *pipeline* eksperimen reproduksibel untuk menjawab dua rumusan masalah utama tanpa kebocoran data (*data leakage*) dan tanpa menggunakan *turn* masa depan (*future turns*):

1. **RQ1 (Pemodelan Lintasan Dialog Eksplisit):** Mengukur peningkatan *Macro-F1* (kelas 1–8 dan 0–8) serta penurunan kesalahan klasifikasi pada kelas yang saling tumpang tindih (**C5 *Neurotic***, **C6 *Obsessional***, dan **C7 *High-Adaptive***) dengan menambahkan model tahap kedua berbasis fitur lintasan (prediksi 1–3 ucapan *seeker* sebelumnya + posisi *turn* absolut) di atas *baseline* Qwen3.
2. **RQ2 (Kalibrasi Logit Terstruktur):** Mengukur apakah kalibrasi logit berparameter ringkas (*structured calibration*) yang dievaluasi menggunakan *nested grouped cross-validation* mampu menyempitkan *generalization gap* antara skor validasi dan skor pengujian dibandingkan pencarian bias acak (*random bias search*).

---

## 2. Spesifikasi Dataset & Kontrak Skema

### 2.1 Sumber dan Pembagian File
* **`train.json` (1.864 sampel):** Digunakan secara eksklusif untuk pelatihan, *5-fold cross-validation*, pemilihan fitur, dan *tuning* kalibrasi.
* **`test.json` (472 sampel):** Dikunci sebagai *held-out test set* yang hanya dievaluasi **satu kali** di tahap akhir tanpa *hyperparameter tuning* apa pun.
* **Distribusi Label (9 Kelas DMRS):**

| Label | Nama Level DMRS | Jumlah Train | Jumlah Test | Proporsi Train |
| :--- | :--- | :---: | :---: | :---: |
| `0` | No Defense | 296 | 75 | 15,88% |
| `1` | Action | 108 | 28 | 5,79% |
| `2` | Major Image-Distorting | 61 | 16 | 3,27% |
| `3` | Disavowal | 99 | 25 | 5,31% |
| `4` | Minor Image-Distorting | 84 | 21 | 4,51% |
| `5` | Neurotic | 48 | 13 | 2,58% |
| `6` | Obsessional | 172 | 44 | 9,23% |
| `7` | High-Adaptive | 968 | 243 | 51,93% |
| `8` | Needs More Info | 28 | 7 | 1,50% |

### 2.2 Kontrak Struktur JSON & Invarian Data
Setiap elemen dalam array JSON memiliki skema berikut:

```json
{
  "id": "train_00002",
  "dialogue_id": "ESConv_001",
  "dialogue": [
    {"text": "hi", "speaker": "seeker"},
    {"text": "Hi. How's it goin'?...", "speaker": "supporter"},
    {"text": "going smooth, i need to get...", "speaker": "seeker"}
  ],
  "current_text": "going smooth, i need to get...",
  "label": 2
}
```

**Aturan Validasi Otomatis (*Assertions*) saat Pemuatan Data:**
1. `dialogue[-1]["speaker"] == "seeker"` (hanya ucapan *seeker* yang dilabeli).
2. `dialogue[-1]["text"].strip() == current_text.strip()` (elemen terakhir pada `dialogue` selalu identik dengan `current_text`).
3. Setiap baris adalah *causal prefix* yang berhenti tepat di ucapan target; tidak ada elemen setelah indeks target (*no future turns*).
4. Total `dialogue_id` unik adalah 200 dialog, dengan 189 `dialogue_id` di `test.json` juga muncul pada potongan giliran lain di `train.json`.

---

## 3. Arsitektur Pencegahan *Data Leakage* (5 Titik Kontrol Wajib)

Notebook wajib mengimplementasikan mekanisme penguncian berikut:

1. **Penetapan Fold Permanen (`fold_assignment.csv`):**
   * Jalankan `StratifiedGroupKFold(n_splits=5)` pada `train.json` menggunakan `groups=dialogue_id` dan `y=label`.
   * Simpan pemetaan `id -> fold_id` (`0..4`) dan `dialogue_id -> fold_id` ke disk sebelum pelatihan dimulai. Seluruh tahap berikutnya wajib memakai pembagian *fold* yang sama persis.
2. **Konstruksi *Prompt* Kausal:**
   * Riwayat dialog di dalam *prompt* hanya mengambil `dialogue[:-1]` (maksimal 30 *turn* terakhir) dan memisahkan `current_text` sebagai target.
3. **Ekstraksi Fitur Historis Berbasis OOF Murni:**
   * Untuk memprediksi ucapan ke-$t$ dalam sebuah dialog, fitur probabilitas ucapan *seeker* sebelumnya ($t-1, t-2, t-3$) **wajib** diambil dari prediksi *out-of-fold* (OOF) yang dihitung menggunakan konteks yang berhenti tepat di $(t-1)$, $(t-2)$, dan $(t-3)$. Dilarang keras memakai label *ground-truth* atau prediksi *in-sample*.
4. **Perutean Model *Held-Out* pada Evaluasi `test.json`:**
   * Karena 189 dialog di `test.json` memiliki ucapan lain di `train.json` (contoh: `ESConv_001` memiliki `train_00001` dan `train_00002` di *train*, lalu `test_00001` di *test*), prediksi untuk ucapan uji dari dialog $D$ **hanya** menggunakan model dari satu *fold* $k$ di mana dialog $D$ berstatus *held-out* (tidak ikut dilatih pada *fold* $k$), baik untuk ucapan target maupun untuk menghitung fitur historis ($t-1, t-2, t-3$) yang mungkin berada di `train.json`.
   * *Pembanding Tambahan:* Hitung juga skor menggunakan rata-rata 5 *fold* standar untuk mengukur besaran inflasi skor akibat irisan dialog pada split resmi.
5. **Pemisahan Tuning Kalibrasi (*Nested Grouped CV*):**
   * Seluruh parameter kalibrasi Tahap 3 dicari pada *inner fold* dan dievaluasi pada *outer fold* berbasis `dialogue_id`.

---

## 4. Spesifikasi Fungsional per Modul Notebook

### Modul 0: Konfigurasi, Reproduksibilitas & *Checkpointing*
* **Konfigurasi Terpusat (`CFG` dataclass/dict):** Menyimpan *random seeds* (mis. `[42, 2026, 777]`), nama model (`Qwen/Qwen3-4B` untuk ablasi cepat, `Qwen/Qwen3-8B` untuk eksperimen utama), parameter QLoRA, *learning rate*, dan *path* penyimpanan.
* **Resumable Execution:** Karena GPU disewa di Vast.ai, setiap *fold* yang selesai dilatih wajib langsung menyimpan artefak (`.npy` / `.parquet` / adaptor LoRA) ke disk dan melewati *fold* yang sudah selesai jika notebook dijalankan ulang.

### Modul 1: Studi Pendahuluan & EDA (Eksekusi CPU)
* Audit jumlah sampel, distribusi 9 kelas pada `train.json`, dan verifikasi 189 irisan `dialogue_id` antara `train.json` dan `test.json`.
* **Analisis Lintasan Empiris:** Urutkan `train.json` berdasarkan `(dialogue_id, len(dialogue))` lalu hitung matriks transisi kelas $9 \times 9$ dari ucapan *seeker* $y_{t-1}$ ke $y_t$, serta distribusi `len(dialogue)` per kelas untuk memvalidasi motivasi RQ1.
* **Audit Panjang Token:** Tokenisasi seluruh `train.json` menggunakan *tokenizer* Qwen3 untuk memverifikasi persentase sampel yang tercakup pada batas `max_length = 1024` (atau `1536`) tanpa memotong `current_text`.
* **Sanity Baseline CPU:** Latih TF-IDF + L2 Logistic Regression (dengan `class_weight="balanced"`) di bawah `StratifiedGroupKFold` 5-fold sebagai patokan batas bawah (*lower bound*).

### Modul 2: Tahap 1 — *Baseline Fine-Tuning* Qwen3 (QLoRA)
* **Arsitektur Model:** Memuat `Qwen/Qwen3-4B` atau `Qwen/Qwen3-8B` sebagai `AutoModelForSequenceClassification` (`num_labels=9`) dengan kuantisasi **4-bit NF4 + *double quantization*** dan presisi komputasi `bfloat16`.
* **Konfigurasi LoRA:** `r = 128`, `lora_alpha = 256`, `lora_dropout = 0.1`, `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "score"]`. Tanpa augmentasi data.
* **Fungsi Objektif & Optimasi:**
  * Bobot kelas *inverse-square-root*:
    $$w_c = \frac{1/\sqrt{n_c}}{\sum_{i=0}^{8} 1/\sqrt{n_i}}$$
    dihitung secara terpisah di dalam setiap *training fold*.
  * *Cross-Entropy Loss* berbobot dengan *label smoothing* $\epsilon = 0.05$.
  * Optimizer `AdamW`, *learning rate* $1.2 \times 10^{-4}$, *cosine annealing scheduler* dengan *warmup* 8%, *weight decay* `0.01`, *gradient clipping* `0.3`, *effective batch size* `16`, hingga `10` *epoch* per *fold*. Simpan *checkpoint* terbaik berdasarkan *Macro-F1* validasi kelas 1–8.
* **Format *Prompt* Input:**
  * **Bagian 1:** Instruksi peran & definisi ringkas 9 kelas DMRS (`0` sampai `8`).
  * **Bagian 2:** Riwayat dialog dari `dialogue[:-1]` (dibatasi maksimal 30 *turn* terakhir dengan awalan `SEEKER:` dan `SUPPORTER:`; jika kosong, isi `[Start of conversation]`).
  * **Bagian 3:** Informasi posisi `(Turn {len(dialogue)})` dan ucapan target `SEEKER: {current_text}`.
* **Keluaran Wajib per Fold:**
  * `oof_logits_s1` ($1864 \times 9$) dan `oof_probs_s1` ($1864 \times 9$).
  * `oof_hidden_states_s1` ($1864 \times d_{\text{hidden}}$) dari token terakhir sebelum *classification head*.
  * Logit dan probabilitas inferensi per *fold* pada seluruh *prefix* yang dibutuhkan untuk `test.json`.

### Modul 3: Tahap 2 — Klasifikasi Sekuensial Berbasis Konteks Dialog (RQ1)
* **Rekonstruksi Urutan Dialog:** Gabungkan `train.json` dan `test.json` hanya untuk pemetaan indeks urutan `(dialogue_id, len(dialogue))` (tanpa menyentuh kolom `label` milik `test.json`), lalu hitung urutan ucapan *seeker* ke-$m$ (`seeker_turn_idx = 0, 1, 2, ...`) dalam setiap dialog.
* **Rekayasa Fitur Lintasan (*Trajectory Features*):**
  1. **Logit/Probabilitas Saat Ini ($t$):** 9 nilai dari keluaran Tahap 1 pada ucapan target.
  2. **Lintasan Historis ($t-1, t-2, t-3$):** 9 probabilitas/logit dari 1 hingga 3 ucapan *seeker* sebelumnya pada `dialogue_id` yang sama (diisi `NaN` jika belum ada giliran sebelumnya; sumber nilai wajib dari OOF Tahap 1 saat CV, dan dari model *held-out fold* saat menguji `test.json`).
  3. **Indeks Temporal:** `abs_turn_idx` (`len(dialogue)`), `seeker_turn_idx`, dan `turn_gap` (selisih *turn* absolut dengan ucapan *seeker* sebelumnya).
  4. **Skor Kematangan DMRS (*Expected Maturity Trajectory*):**
     $$E_t = \sum_{c=1}^{7} c \cdot P_t(c \mid c \in \{1..7\})$$
     Beserta perubahan kematangan $\Delta E_t = E_t - E_{t-1}$ dan rata-rata kumulatif $\bar{E}_{<t}$ untuk menangkap pergeseran menuju/menjauh dari *High-Adaptive* (C7).
  5. **Sinyal Ketidakpastian & Konteks Supporter:** Entropi prediksi saat ini $H(P_t)$, margin probabilitas peringkat-1 vs peringkat-2, selisih $P_t(C7) - P_t(C6)$, panjang kata ucapan *seeker* saat ini, serta panjang kata ucapan *supporter* tepat sebelumnya (`dialogue[-2]`) dan indikator tanda tanya `?`.
* **Model Sekuensial yang Dilatih (di bawah 5-fold `fold_id` yang sama):**
  * **LightGBM** dan **CatBoost** dengan pembatasan kompleksitas ketat untuk mencegah *overfitting* pada $N = 1864$ (`max_depth` $\in \{2, 3\}$, `min_child_samples` $\in \{20, 30\}$, regularisasi L1/L2, pembobotan kelas seimbang).
  * **Pembanding Linear Sekuensial:** L2-regularized Multinomial Logistic Regression di atas fitur lintasan yang sama (untuk menguji apakah hubungan transisi bersifat linear atau non-linear).
* **Keluaran Tahap 2:** Logit mentah (*raw margins*, sebelum *softmax*) dari model sekuensial terbaik (`oof_logits_s2` dan `test_logits_s2`) sebagai masukan Tahap 3.

### Modul 4: Tahap 3 — Kalibrasi Logit Terstruktur (RQ2)
* **Protokol Validasi:** *Nested Grouped Cross-Validation* berbasis `dialogue_id` (5 *outer fold* $\times$ 4 *inner fold*) pada matriks logit OOF.
* **Kandidat Metode Kalibrasi Terstruktur (Parameter Ringkas):**
  1. **Prior Logit Adjustment (1 Parameter $\tau$):**
     $$z'_c = z_c - \tau \log \pi_c$$
     dengan $\pi_c$ dihitung hanya pada *inner training fold*.
  2. **Temperature Scaling + Prior Adjustment (2 Parameter $T, \tau$):**
     $$z'_c = \frac{z_c}{T} - \tau \log \pi_c$$
  3. **Structured C0/C7 Gate + Prior Adjustment (3 Parameter $b_0, b_7, \tau$):** Menambahkan penyesuaian terpisah untuk kelas yang paling terpisah (`C0`) dan kelas mayoritas yang mendominasi kesalahan (`C7`), sementara kelas minoritas lain diatur secara terstruktur oleh $\tau$.
  4. **L2-Regularized Logistic Regression / Platt Scaling Multikelas** pada logit OOF.
* **Baseline Pembanding RQ2:** Implementasi ulang **Random Bias Search (22.000 percobaan acak $\delta_0 \dots \delta_8$)** ala LinguIUTics pada *inner fold* yang sama.
* **Metrik Utama RQ2:** Selisih (*generalization gap*) antara rata-rata skor *Macro-F1* di *inner-CV* dan skor *Macro-F1* di *outer-fold*:
  $$\Delta_{\text{gap}} = \text{F1}_{\text{inner}} - \text{F1}_{\text{outer}}$$

### Modul 5: Evaluasi Ablasi, Analisis Error & Pengujian Akhir
* **Tabel Ablasi Utama (dilaporkan pada Grouped CV dan Held-Out Test):**
  1. `Baseline`: Qwen3-8B QLoRA (`argmax` standar)
  2. `Baseline + Structured Calibration`: Isolasi kontribusi RQ2 tanpa tahap sekuensial
  3. `Baseline + Sequential Stage (RQ1)`: Kontribusi fitur lintasan dialog tanpa kalibrasi
  4. `Baseline + Sequential Stage + Structured Calibration (Full)`: *Pipeline* lengkap
* **Analisis Diagnostik Mendalam:**
  * *Macro-F1* kelas 1–8 (metrik resmi) dan kelas 0–8.
  * *Precision, Recall, F1* per kelas (fokus pada pengurangan kebingungan **C5/C6/C7** dan pemulihan kelas minoritas **C2, C5, C8**).
  * *Normalized Confusion Matrix* sebelum dan sesudah tahap sekuensial serta kalibrasi.
  * *Bootstrap Confidence Interval* 95% (1.000 iterasi *resampling*) dan varians antar-*seed*.
  * Pengukuran inflasi *split*: selisih skor `test.json` antara perutean *held-out fold* vs rata-rata 5 *fold* standar.

---

## 5. Struktur Direktori Artefak & Kriteria Keberhasilan

```text
artifacts/
├── splits/
│   └── train_folds.parquet          # id, dialogue_id, fold_id, seeker_turn_idx, abs_turn_idx
├── stage1_qwen3_8b/
│   ├── seed_42/
│   │   ├── oof_logits.npy           # (1864, 9)
│   │   ├── oof_hidden.npy           # (1864, hidden_dim)
│   │   ├── test_logits_by_fold.npy  # (5, 472, 9) -> untuk held-out routing & 5-fold avg
│   │   └── fold_{0..4}_best/        # LoRA adapter checkpoints
├── stage2_sequential/
│   ├── oof_logits_lgbm.npy          # (1864, 9)
│   ├── oof_logits_catboost.npy      # (1864, 9)
│   └── test_logits_routed.npy       # (472, 9)
└── results/
    ├── ablation_summary.csv
    ├── per_class_metrics.csv
    └── confusion_matrices.png
```

### Kriteria Penerimaan (*Acceptance Criteria*)
1. **Zero Leakage Verified:** Unit test otomatis memastikan `len(set(train_fold_dialogues) & set(val_fold_dialogues)) == 0` di seluruh 5 *fold* dan 20 *inner-nested fold*.
2. **Causal Integrity Verified:** Tidak ada fitur pada baris ke-$t$ yang berkorelasi atau mengakses data dari indeks $> t$ pada `dialogue_id` yang sama.
3. **VRAM Compliance:** Puncak penggunaan VRAM pada Tahap 1 tidak melebihi $22\text{ GB}$ pada GPU $24\text{ GB}$ (menggunakan *gradient checkpointing*, *paged AdamW 8-bit*, dan *dynamic padding* kelipatan 8).
4. **Reproducibility:** Seluruh tabel dan grafik di Bab 4 skripsi dapat dihasilkan ulang secara otomatis dari artefak `.npy` dan `.parquet` yang tersimpan tanpa perlu melatih ulang Qwen3.