# Klasifikasi Kategori Produk E-Commerce: ANN vs Naive Bayes vs SVM vs Random Forest

Perbandingan empat algoritma klasifikasi teks untuk memprediksi kategori produk dari deskripsinya, memakai pipeline TF-IDF yang sama, data yang sudah dideduplikasi, dan tanpa kebocoran data (*data leakage*).

## Ringkasan

| Item | Keterangan |
|---|---|
| Tugas | Klasifikasi teks multikelas (4 kategori) |
| Kategori | Household, Books, Electronics, Clothing & Accessories |
| Dataset | E-commerce Text Dataset (Gautam, Zenodo; tersedia di Kaggle sebagai *Ecommerce Text Classification*) |
| Fitur | TF-IDF unigram + bigram, 20.000 fitur |
| Model | ANN (`MLPClassifier`), Multinomial Naive Bayes, Linear SVM (`LinearSVC`), Random Forest |
| Metrik utama | F1-macro (kelas tidak seimbang, rasio 1,99 : 1) |
| Seed | `42` untuk semua proses acak |

**Temuan utama:** Linear SVM memberi kombinasi terbaik antara akurasi, kecepatan, dan interpretabilitas. ANN hampir setara dalam akurasi tetapi sekitar 7 kali lebih lama dilatih.

## Hasil pada Data Uji (5.561 dokumen)

| Model | Accuracy | F1-macro | Waktu training (s) | Waktu prediksi (s) |
|---|---|---|---|---|
| **SVM** | **0,9603** | **0,9601** | 10,90 | 0,753 |
| ANN | 0,9581 | 0,9580 | 78,28 | 0,822 |
| Naive Bayes | 0,9439 | 0,9440 | 9,22 | 0,699 |
| Random Forest | 0,9324 | 0,9322 | 156,58 | 1,842 |

Hyperparameter terbaik hasil grid search: ANN `alpha=0.001`, Naive Bayes `alpha=0.01`, SVM `C=1`, Random Forest `n_estimators=300` dan `max_depth=None`. Tuning hanya menaikkan F1-macro paling banyak 0,0042 (Naive Bayes), sehingga peringkat model tidak berubah.

Sebanyak 146 dokumen uji (2,63%) salah diklasifikasi oleh keempat model sekaligus. Kesalahan ini kebanyakan berupa kebingungan dengan kategori Household, dan berkaitan dengan produk ambigu, deskripsi pendek, atau label yang kemungkinan keliru.

## Dataset

Dataset diunduh dari Kaggle (*Ecommerce Text Classification*), lalu letakkan sebagai `ecommerceDataset.csv` di folder yang sama dengan notebook.

File tidak memiliki header: kolom pertama adalah label kategori, kolom kedua adalah teks deskripsi produk. Jumlah data mentah 50.425 baris.

Hasil pembersihan data:

| Tahap | Jumlah data |
|---|---|
| Data awal | 50.425 |
| Setelah hapus teks kosong dan label konflik | 50.424 |
| Setelah hapus duplikat | 27.802 |

Sebanyak 22.622 baris (44,86%) adalah duplikat persis. Menghapusnya sebelum pembagian data penting agar teks yang sama tidak muncul di data latih dan data uji sekaligus.

## Alur Eksperimen

Notebook dijalankan berurutan dan mengikuti tahap berikut:

1. **Pengumpulan data**: membaca CSV tanpa header.
2. **Pembersihan data**: hapus teks kosong, hapus label konflik, hapus duplikat.
3. **EDA**: distribusi kelas, panjang teks per kategori, kata teratas per kategori.
4. **Preprocessing teks**: *case folding*, hapus URL/angka/tanda baca, tokenisasi, hapus *stopword* dan token ≤ 2 huruf, lemmatisasi WordNet.
5. **Pembagian data**: stratified 80:20 (22.241 latih, 5.561 uji). Data uji tidak disentuh sampai evaluasi akhir.
6. **Ekstraksi fitur**: TF-IDF (`ngram_range=(1,2)`, `max_features=20000`, `min_df=2`, `sublinear_tf=True`) di dalam `Pipeline`, sehingga di tiap fold CV vocabulary dan IDF hanya dipelajari dari data latih fold tersebut.
7. **Modeling**: tiap algoritma dilatih dengan hyperparameter default, lalu di-tuning dengan Grid Search + stratified CV (metrik F1-macro).
8. **Evaluasi**: accuracy, precision/recall/F1 macro, waktu training dan prediksi, *confusion matrix*.
9. **Interpretasi model**: kata penentu tiap kategori dan kemiripan antar model (Jaccard).
10. **Analisis error**: dokumen yang salah diklasifikasi oleh keempat model, dikelompokkan secara heuristik.
11. **Kesimpulan**: ringkasan angka otomatis untuk paper.

### Ruang pencarian hyperparameter

| Model | Pengaturan tetap | Grid | CV |
|---|---|---|---|
| ANN (`MLPClassifier`) | 1 hidden layer 128 neuron, ReLU, Adam, maks. 30 epoch, *early stopping* | `alpha` ∈ {0,0001; 0,001} | 3-fold |
| Multinomial Naive Bayes | none | `alpha` ∈ {0,01; 0,1; 0,5; 1,0} | 5-fold |
| Linear SVM (`LinearSVC`) | one-vs-rest, `max_iter=10000` | `C` ∈ {0,01; 0,1; 1; 10} | 5-fold |
| Random Forest | kriteria Gini | `n_estimators` ∈ {100; 300}, `max_depth` ∈ {None; 50} | 5-fold |

### Metode interpretasi

| Model | Cara menentukan kata penting |
|---|---|
| ANN | *Connection weight* (Olden et al., 2004): bobot input→hidden dikali hidden→output, dijumlah atas 128 neuron. Bersifat pendekatan karena ReLU dan bias diabaikan. |
| SVM | Koefisien positif tertinggi per kelas (skema one-vs-rest) |
| Naive Bayes | Peluang P(kata \| kategori) tertinggi |
| Random Forest | *Mean decrease in impurity*, 20 kata teratas secara global |

## Cara Menjalankan

### 1. Prasyarat

- Python 3.9 atau lebih baru
- Disarankan RAM ≥ 8 GB (TF-IDF 20.000 fitur dengan ANN dan Random Forest cukup berat)

### 2. Instalasi

```bash
git clone <url-repositori-anda>
cd <nama-folder>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install pandas numpy scipy scikit-learn nltk matplotlib seaborn jupyter
```

Data NLTK (`stopwords`, `wordnet`, `omw-1.4`) diunduh otomatis oleh notebook pada sel import.

### 3. Menjalankan notebook

```bash
jupyter notebook klasifikasi_ecommerce_ann_fix.ipynb
```

Pastikan `ecommerceDataset.csv` sudah ada di folder yang sama, lalu jalankan semua sel secara berurutan (*Run All*).
