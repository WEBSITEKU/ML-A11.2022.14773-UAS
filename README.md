# Prediksi penyakit jantung menggunakan Logistic Regression
### Permasalahan dan Ringkasan project
Penyakit jantung adalah salah satu penyebab utama kematian di seluruh dunia. Diagnosis dini dan akurat sangat penting untuk penanganan yang efektif. Namun, diagnosis manual seringkali memakan waktu dan dapat berisiko terhadap kesalahan. Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi apakah seorang pasien menderita penyakit jantung berdasarkan berbagai fitur seperti usia, jenis kelamin, tekanan darah, kolesterol, dan lain-lain. Model ini diharapkan dapat membantu dokter dalam mendiagnosis penyakit jantung dengan lebih cepat dan akurat.
### Tujuan yang akan dicapai:
1. Membangun model machine learning yang dapat memprediksi penyakit jantung dengan akurasi yang tinggi.
2. Mengevaluasi kinerja model dan membuat rekomendasi untuk perbaikan.
### Model / alur penyelesaian:
![Load Dataset Awal kecil](https://github.com/user-attachments/assets/c2c0cbf7-9387-462c-a8e3-64f83490d8c3)
### Penjelasan Dataset, EDA, dan Proses features dataset
Dataset yang digunakan dalam penelitian ini berasal dari situs web Kaggle, yang merupakan platform populer untuk kompetisi machine learning dan dataset. Dataset ini bersifat publik, yang berarti siapa saja dapat mengakses dan menggunakannya untuk tujuan penelitian. Dataset ini berisi data historis pasien yang terkena penyakit jantung dan yang tidak berdasarkan 13 atribut dan 1 label dalam dataset yang berkaitan dengan kondisi jantung.
Berikut adalah penjelasan 13 atribut pada dataset :
- age: umur dalam tahun
- sex: 1 = laki-laki; 0 = perempuan
- CP (Chest Pain): jenis nyeri dada
  - Value 0: angina tipikal
  - Value 1: angina atipikal
  - Value 2: nyeri non-anginal
  - Value 3: tanpa gejala
- Trestbps: tekanan darah istirahat (mm Hg saat masuk rumah sakit)
- Chol: kolesterol serum dalam mg/dl
- FBS (Fasting Blood Sugar): kadar gula darah puasa > 120 mg/dl (1 = ya; 0 = tidak)
- Restecg: hasil elektrokardiografi istirahat
  - Value 0: normal
  - Value 1: abnormalitas gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST > 0.05 mV)
  - Value 2: hipertrofi ventrikel kiri mungkin atau pasti menurut kriteria Estes
- Thalach: detak jantung maksimum yang dicapai
- Exang (Exercise induced angina): angina yang diinduksi olahraga (1 = ya; 0 = tidak)
- Oldpeak: depresi ST yang diinduksi oleh latihan relatif terhadap istirahat
- Slope: kemiringan segmen ST puncak latihan paling tinggi
  - Value 0: naik
  - Value 1: datar
  - Value 2: menurun
- CA: jumlah pembuluh besar (0-3) yang berwarna dengan fluoroskopi
- Thal: 0 = normal; 1 = cacat tetap; 2 = cacat reversibel
- condition: 0 = tidak ada penyakit, 1 = penyakit

. Variabel variable ini digunakan sebagai fitur dalam model regresi logistik. Selain itu, dataset ini juga mencakup label ‘condition’, yang menentukan apakah pasien terkena penyakit atau tidak.
### Proses Learning dan Modeling
proses pembuatan dan evaluasi model Logistic Regression untuk memprediksi penyakit jantung menggunakan dataset `heart_cleveland_upload.csv`. Berikut penjelasan langkah demi langkah dari kode tersebut:

1. **Import Library**: Library yang diperlukan seperti numpy, pandas, train_test_split, LogisticRegression, dan classification_report dari scikit-learn diimpor.

2. **Membaca Data**: Data tentang penyakit jantung dibaca dari file CSV `heart_cleveland_upload.csv` menggunakan pandas dan disimpan dalam variabel `heart_data`.

3. **Memisahkan Fitur dan Label**: Data dipisahkan menjadi fitur (`X`) dan label (`Y`) dengan `X` adalah semua kolom kecuali kolom 'condition' dan `Y` adalah kolom 'condition'.

4. **Mencetak Label**: Label `Y` dicetak untuk melihat nilai target.

5. **Membagi Data**: Data dibagi menjadi data pelatihan dan pengujian dengan proporsi 80:20 menggunakan `train_test_split`, dengan `stratify` diatur untuk menjaga distribusi kelas yang sama antara data pelatihan dan pengujian, dan `random_state` untuk memastikan pembagian yang sama setiap kali kode dijalankan.

6. **Mencetak Bentuk Data**: Bentuk dari keseluruhan data, data pelatihan, dan data pengujian dicetak untuk memverifikasi pemisahan yang benar.

7. **Membuat dan Melatih Model**: Model Logistic Regression dibuat dan dilatih menggunakan data pelatihan (`X_train` dan `Y_train`).

8. **Evaluasi Model pada Data Pengujian**: Model memprediksi label pada data pengujian (`X_test`) dan laporan klasifikasi (classification report) dihasilkan, yang mencakup precision, recall, dan F1-score untuk kedua kelas ('Tidak Terkena Penyakit Jantung' dan 'Terkena Penyakit Jantung').

9. **Prediksi Data Baru**: Sebuah contoh data baru `(61, 1, 0, 134, 234, 0, 0, 145, 0, 2.6, 1, 2, 0)` diberikan untuk diprediksi oleh model. Data ini diubah menjadi array numpy dan direshape agar sesuai dengan format input model. Model kemudian memprediksi apakah pasien tersebut terkena penyakit jantung atau tidak berdasarkan data input ini.

10. **Mencetak Hasil Prediksi**: Hasil prediksi dicetak. Jika prediksi adalah 0, maka pasien tidak terkena penyakit jantung, jika 1, maka pasien terkena penyakit jantung.

Secara keseluruhan, kode ini melakukan proses mulai dari pembacaan data, pemisahan data, pelatihan model, evaluasi performa, hingga prediksi pada data baru. Dengan menggunakan Logistic Regression, model berhasil memprediksi risiko penyakit jantung dan memberikan laporan klasifikasi yang menunjukkan performa model.

### Performa Model
Berikut adalah hasil dari evaluasi modelnya:

![image](https://github.com/user-attachments/assets/046aa55e-2ee6-4cf2-a9c4-224f773e4dd7)


1. Precision:
Tidak Terkena Penyakit Jantung: 0.88 (88% dari prediksi tidak terkena penyakit jantung adalah benar)
Terkena Penyakit Jantung: 0.92 (92% dari prediksi terkena penyakit jantung adalah benar)

2. Recall:
Tidak Terkena Penyakit Jantung: 0.94 (94% dari kasus yang benar-benar tidak terkena penyakit jantung berhasil terdeteksi)
Terkena Penyakit Jantung: 0.86 (86% dari kasus yang benar-benar terkena penyakit jantung berhasil terdeteksi)

3. F1-Score:
Tidak Terkena Penyakit Jantung: 0.91 (Harmonic mean dari precision dan recall)
Terkena Penyakit Jantung: 0.89

4. Akurasi: 90% (90% dari keseluruhan prediksi model adalah benar)

5. Macro avg: Rata-rata tidak berbobot dari precision, recall, dan F1-score untuk kedua kelas.

6. Weighted avg: Rata-rata berbobot dari precision, recall, dan F1-score, mempertimbangkan jumlah sampel di setiap kelas.

#### Kesimpulan Performanya:
1. **Akurasi Tinggi**: Dengan akurasi 90%, model memiliki kemampuan yang baik dalam memprediksi penyakit jantung.
2. **Precision Tinggi**: Menunjukkan bahwa model jarang memberikan false positives. Prediksi positif model (baik terkena maupun tidak terkena penyakit jantung) sangat akurat.
3. **Recall Tinggi**: Menunjukkan bahwa model mampu mendeteksi sebagian besar kasus yang sebenarnya, baik terkena maupun tidak terkena penyakit jantung.
4. **F1-Score Tinggi**: Mengindikasikan keseimbangan yang baik antara precision dan recall untuk kedua kelas.

#### Rekomendasi:
- **Model Validation**: Pastikan model divalidasi dengan data yang berbeda untuk memastikan generalisasi.
- **Hyperparameter Tuning**: Pertimbangkan untuk melakukan tuning hyperparameter untuk meningkatkan performa lebih lanjut.
- **Feature Engineering**: Lakukan analisis lebih lanjut pada fitur untuk melihat apakah ada fitur yang dapat dioptimalkan atau fitur baru yang bisa ditambahkan.
- **Cross-Validation**: Gunakan teknik cross-validation untuk memastikan stabilitas dan konsistensi performa model.

Secara keseluruhan, model ini menunjukkan performa yang sangat baik dalam memprediksi kondisi penyakit jantung dan dapat menjadi alat yang berharga untuk diagnosis awal. Hasil prediksi untuk data baru menunjukkan bahwa model dapat mengidentifikasi pasien yang berisiko terkena penyakit jantung dengan tingkat kepercayaan yang tinggi.

### Hasil dan Kesimpulan
Disini saya menambahkan untuk prediksi model Logistic Regression, 

#### Langkah-langkah Prediksi

1. **Input Data**: Contoh data pasien yang akan diprediksi:
   untuk data input yang diberikan:
   ```python
   input_data = (61, 1, 0, 134, 234, 0, 0, 145, 0, 2.6, 1, 2, 0)
   ```

   Artinya, fitur-fitur yang digunakan untuk prediksi adalah:
   - Usia: 61
   - Jenis Kelamin: 1 (male)
   - CP (chest pain type): 0 (typical angina)
   - Trestbps (resting blood pressure): 134
   - Chol (serum cholesterol): 234
   - Fbs (fasting blood sugar > 120 mg/dl): 0
   - Restecg (resting electrocardiographic results): 0
   - Thalach (maximum heart rate achieved): 145
   - Exang (exercise induced angina): 0
   - Oldpeak (ST depression induced by exercise relative to rest): 2.6
   - Slope (the slope of the peak exercise ST segment): 1
   - Ca (number of major vessels colored by fluoroscopy): 2
   - Thal (thalassemia): 0

3. **Mengubah Data Menjadi Numpy Array**:
   ```python
   input_data_as_numpy_array = np.asarray(input_data)
   ```

   Data input yang awalnya berbentuk tuple diubah menjadi numpy array untuk kompatibilitas dengan model prediksi.

4. **Reshape Data**:
   ```python
   input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
   ```

   Data diubah bentuknya menjadi satu baris dengan beberapa kolom, yang merupakan format input yang diterima oleh model. `reshape(1, -1)` berarti satu sampel dengan jumlah kolom yang sesuai dengan jumlah fitur (13 fitur).

5. **Melakukan Prediksi**:
   ```python
   prediction = model.predict(input_data_reshaped)
   ```

   Model Logistic Regression yang telah dilatih menggunakan data sebelumnya akan memprediksi label kondisi pasien berdasarkan data input yang diberikan.

6. **Mencetak Hasil Prediksi**:
   ```python
   print(prediction)

   if (prediction[0] == 0):
       print('pasien tidak terkena penyakit jantung')
   else:
       print('pasien terkena penyakit jantung')
   ```

   Hasil prediksi dicetak:
   - Jika `prediction[0]` adalah 0, maka model memprediksi bahwa pasien **tidak terkena penyakit jantung**.
   - Jika `prediction[0]` adalah 1, maka model memprediksi bahwa pasien **terkena penyakit jantung**.

#### Kesimpulan
Model Logistic Regression digunakan untuk memprediksi apakah seorang pasien terkena penyakit jantung berdasarkan beberapa fitur seperti usia, jenis kelamin, tekanan darah, kolesterol, dan lainnya. Berdasarkan data input yang diberikan (contoh pasien berusia 61 tahun dengan parameter kesehatan yang dijelaskan di atas), model menghasilkan prediksi yang menunjukkan apakah pasien tersebut terkena penyakit jantung atau tidak.

Misalnya, jika hasil prediksi adalah `[1]`, maka model menunjukkan bahwa pasien tersebut terkena penyakit jantung. Hasil ini berguna untuk membantu dokter dalam membuat keputusan klinis lebih lanjut, meskipun selalu disarankan untuk melakukan pemeriksaan medis lebih lanjut untuk konfirmasi.
