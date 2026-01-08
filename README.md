# Analisis Sentimen Ulasan Tokopedia (Google Play Store)

Proyek ini berisi 2 notebook:
1) **`notebook_scraping.ipynb`**: scraping ulasan aplikasi **Tokopedia** dari Google Play Store menggunakan `google-play-scraper`.
2) **`notebook_model.ipynb`**: preprocessing teks, pelabelan sentimen berbasis **lexicon**, lalu pelatihan model **Deep Learning (LSTM, CNN, GRU)** untuk klasifikasi sentimen.

---

## Struktur Proyek

```
.
├── notebook_scraping.ipynb
├── notebook_model.ipynb
├── requirements.txt
└── (output saat dijalankan)
    ├── ulasan_aplikasi_tokopedia_15K.csv
    ├── model_LSTM.h5
    ├── model_CNN.h5
    └── model_GRU.h5
```

> Catatan: file output di atas dibuat setelah notebook dijalankan.

---

## 1) Scraping Dataset (`notebook_scraping.ipynb`)

Notebook scraping mengambil ulasan aplikasi Tokopedia dengan konfigurasi utama:
- **App ID**: `com.tokopedia.tkpd`
- **Bahasa/Negara**: `lang='id'`, `country='id'`
- **Urutan**: `Sort.MOST_RELEVANT`
- **Target jumlah**: `count=30000`
- **Filter rating**: `filter_score_with=5` (hanya rating bintang 5)
- Output disimpan ke **CSV**: `ulasan_aplikasi_tokopedia_15K.csv`

### Jalankan
Buka `notebook_scraping.ipynb` lalu jalankan semua cell.

---

## 2) Pemodelan Sentimen (`notebook_model.ipynb`)

### A. Load & Cleaning Data
- Membaca dataset dari `ulasan_aplikasi_tokopedia_15K.csv`
- Menghapus missing value pada kolom `content` dan menghapus duplikasi.

### B. Preprocessing Teks
Pipeline preprocessing yang diterapkan pada kolom `content`:
1. **Cleaning**: hapus mention, hashtag, RT, URL, angka, tanda baca, dan rapikan spasi.
2. **Casefolding**: ubah ke huruf kecil.
3. **Normalisasi slangwords**: ubah kata gaul menjadi kata baku (menggunakan kamus slang di notebook).
4. **Tokenizing**: memecah teks menjadi token.
5. **Stopword removal**: menghapus stopwords (Indonesia & Inggris).
6. **Join kembali**: token digabung jadi teks akhir (`text_akhir`).

> Di notebook juga ada fungsi stemming (Sastrawi), tetapi disebutkan tidak dipakai karena prosesnya lama.

### C. Pelabelan Data (Lexicon-Based)
Label sentimen dibuat otomatis memakai kamus **lexicon positif & negatif** (diunduh dari GitHub lewat `requests`), kemudian dihitung skor sentimen dengan menjumlahkan bobot kata.
Aturan label (berdasarkan skor):
- **positive**: `score >= 0`
- **negative**: `score <= -7`
- **neutral**: selain itu

Hasil label disimpan pada kolom:
- `polarity_score`
- `polarity`

### D. Tokenisasi untuk Model
- `Tokenizer(num_words=2500)` lalu `texts_to_sequences` + `pad_sequences`
- Label diubah menjadi numerik via `LabelEncoder`, lalu one-hot via `to_categorical`.

### E. Model yang Dilatih
Notebook membangun dan melatih 3 model:

**1) LSTM**
- Embedding(2500, 256)
- 2 layer LSTM + Dense
- Optimizer Adam (lr=0.001), loss categorical_crossentropy

**2) CNN**
- Embedding(2500, 512)
- Conv1D + MaxPooling + Flatten + Dense
- Optimizer Adam (lr=0.001)

**3) GRU (Bidirectional)**
- Embedding(2500, 512)
- SpatialDropout1D
- 2 layer Bidirectional GRU + Dense
- Optimizer Adam (lr=0.001)

Notebook memakai callback kustom untuk **menghentikan training** jika `val_accuracy > 0.92`.

### F. Evaluasi & Penyimpanan Model
- Model dievaluasi menggunakan test set (accuracy).
- Model disimpan sebagai:
  - `model_LSTM.h5`
  - `model_CNN.h5`
  - `model_GRU.h5`
- Notebook juga membuat ringkasan `results_df` (akurasi train & test).

### G. Prediksi Data Baru
Di bagian akhir, notebook menyiapkan contoh teks baru, melakukan tokenisasi & padding, lalu memprediksi label dengan ketiga model.

---

## Instalasi

### Opsi A (sesuai file)
Install semua dependency dari `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Opsi B (lebih ringan, biasanya cukup untuk menjalankan notebook)
Karena `requirements.txt` sangat besar, kamu bisa coba minimal dependency berikut:
```bash
pip install google-play-scraper pandas numpy nltk Sastrawi scikit-learn tensorflow matplotlib seaborn wordcloud requests
```

> Jika kamu memakai Opsi B dan ada error import tertentu, tambahkan library yang hilang sesuai pesan error.

---

## Cara Menjalankan (ringkas)

1. **Scraping**  
   Jalankan `notebook_scraping.ipynb` → menghasilkan `ulasan_aplikasi_tokopedia_15K.csv`

2. **Training & Evaluasi**  
   Jalankan `notebook_model.ipynb` → menghasilkan `model_*.h5` + evaluasi

---

## Catatan Penting (untuk interpretasi hasil)
- Dataset hasil scraping **difilter hanya bintang 5** (`filter_score_with=5`). Ini bisa membuat distribusi sentimen “bias” (misalnya dominan positif) dan kurang merepresentasikan keluhan/ulasan negatif.
- Label sentimen dibuat otomatis via lexicon (bukan anotasi manual). Akurasi model tinggi belum tentu berarti generalisasi bagus—karena label “ground truth” juga hasil aturan.

---

## Kredit / Referensi
- Scraping: `google-play-scraper`
- Pelabelan: lexicon positif & negatif diunduh dari GitHub menggunakan `requests`.
