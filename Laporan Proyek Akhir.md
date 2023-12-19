# Laporan Proyek Machine Learning - Adam Adnan

### Project Overview
Meskipun Indonesia memiliki tempat wisata yang menarik - pedalaman yang indah, reruntuhan budaya dan sejarah yang menarik, pantai, dan banyak lagi. Namun masih banyak turis asing maupun lokal yang tidak mengetahui lokasi tempat wisata tersebut.
Masalahnya turis asing maupun lokal akan merasa bosan jika hanya mengetahui satu tempat wisata saja. Akibatnya, Jumlah yang berkunjung akan berkurang.

Dari masalah di atas Bayangkan kita adalah seorang Data Scientist di perusahaan pariwisata. Setelah sekian lama beroperasi, perusahaan kita berhasil mengumpulkan berbagi informasi mengenai pengguna, pariwisata dan objek wisata, serta rating yang diberikan pengguna untuk pariwisata tersebut. Seluruh informasi ini terkumpul dalam _Tourism and destination._
### Business Understanding
Sebagai seorang Data Scientis, untuk kita ingin memanfaatkan data tersebut untuk meningkatkan transaksi di perusahaan. Kembangkan sebuah sistem rekomendasi pariwisata untuk menjawab permasalahan tersebut:
- Dengan data rating yang Anda miliki, bagaimana perusahaan dapat merekomendasikan tujuan wisata lain yang mungkin disukai dan belum pernah dikunjungi oleh pengguna? 

- Menghasilkan sejumlah rekomendasi tujuan wisata yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi sebelumnya dengan teknik collaborative filtering.

### Data Understanding 
Dataset yang digunakan pada proyek ini adalah Indonesia Tourism Destination yang dapat diunduh dari https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/code . Kita akan menggunakan Google Colaboratory untuk membuat model sistem rekomendasi.
Pertama, Kita download dulu dataset. Jika berhasil, file indonesia-tourism-destination.zip akan masuk ke sistem storage Google Colab. Kemudian, lakukan unzip file seperti pada Gambar 1.
[![Screenshot-2022-09-21-at-13-15-10-Google-Colaboratory.png](https://i.postimg.cc/d1CgPpxf/Screenshot-2022-09-21-at-13-15-10-Google-Colaboratory.png)](https://postimg.cc/1n9vDCP0)
Gambar 1. Hasil Unzip file
Pada Gambar 1, terdapat 4 file csv.
Keempat file tersebut dapat kita kategorikan menjadi 3, yaitu Tourism, consumers, dan user-item-rating.
- Tourism 
-- package_tourism.csv
-- tourism_with_id.csv
- Consumers
--user.csv
- User-Item-Rating
--tourism_rating.csv

Dataset yang akan kita gunakan adalah tourism_rating.csv yang dapat kita lihat pada Tabel 1.

Table 1. Data Tourism_Rating 
||User_Id|Place_Id|Place_Ratings|
| --- | --- | --- | --- |
|0|1|179|3|
|1|1|344|2|
|2|1|5|5|
|3|1|373|3|
|4|1|101|4|
|----|----|----|----|
|9995|300|425|2|
|9996|300|64|4|
|9997|300|311|3|
|9997|300|279|3|
|9999|300|163|2|
Perhatikanlah, data rating memiliki 10000 baris dan 3 kolom
**Variabel - variabel pada rating pariwisata**
- User_Id : merepresentasikan Id dari pengguna
- Place_Id : merepresentasikan Id dari tempat wisata
- Place_Ratings : merepresentasikan penilaian yang diberikan pengguna

### Data Preparation
Pada tahap ini, Kita perlu melakukan persiapa data menyandikan(encode) fitur 'User_ID' dan 'Place_Id' ke dalam indeks integer.Kemudian kita petakan User_Id dan Place_Id ke dataframe yang berkaitan. Terakhir, cek beberapa hala dalam data seperti jumlah user,
jumlah tourism, dan mengubah nilai rating menjadi float.
[![Screenshot-2022-09-21-at-11-47-11-Google-Colaboratory.png](https://i.postimg.cc/W15FXXck/Screenshot-2022-09-21-at-11-47-11-Google-Colaboratory.png)](https://postimg.cc/G4smpP2L)
Pada gambar diatas kita telah mengecek beberapa hal dalam data seperti jumlah user = 300, jumlah tourism = 437, kemudian mengubah nilai rating menjadi float.

**Membagi Data untuk Training dan Validasi**
Pada tahap ini kita akan melakukan pembagian data menjadi data training dan validasi. Namun, sebelumnya, acak datanya terlebih dahulu agar distribusinya menjadi random yang ditunjukkan pada Tabel 2.

Tabel 2. Mengacak pada data sampel dengan frac = 1 dan radom_state = 42

| |User_Id|Place_Id|Placee_Ratings|user|tourism|
|---| ----  | ----  |       ---    |---| ---- |
|6252| 188 | 207 | 3.0 | 187 | 262 |
|4684| 142 | 268 | 2.0 | 141 | 83 |
|1731| 54 | 103 | 3.0 | 53 | 10 |
|4742| 144 | 119 | 3.0 | 143 | 141 |
|4521| 138 | 288 | 5.0 | 137 | 283 |
|----|----|----|----|----|----|
|5734|173|402|5.0|172|175|
|5191|157|85|4.0|156|33|
|5390|163|18|3.0|162|52|
|860|28|416|4.0|27|187|
|7270|219|258|2.0|218|6|

Pada Gambar di atas dataset kita telah teracak.
Selanjutnya, kita bagi data train dan validasi dengan komposisi 80:20. Namun sebelumnya, kita perlu memetakan (mapping) data user dan tourism menjadi satu value terlebih dahulu. lalu, buatlah rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training.
[![Screenshot-2022-09-21-at-11-45-13-Google-Colaboratory.png](https://i.postimg.cc/26qt9cq9/Screenshot-2022-09-21-at-11-45-13-Google-Colaboratory.png)](https://postimg.cc/dkYnL9jG)
Data telah siap untuk dimasukkan ke dalam model.
### Modeling
**Proses Training**
Pada tahap ini, model menghitung skor kecocokan antara pengguna dan pariwisata dengan teknik embedding. Pertama, kita melakukan proses embedding terhadapa data pengguna dan pariwisata. Selanjutnya, lakukan operasi perkalian dot product antara embedding pengguna dan pariwisata. Selain itu, kita juga dapat menambahkan bias untuk setiap pengguna dan pariwisat. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

Di sini, kita membuat class RecommerderNet dengan keras Model class. Kode class RecommenderNet ini terinspirasi dari tutorial dalam situs Keras dengan beberapa adaptasi sesuai kasus yang sedang kita selesaikan. Setelah itu kita melakukan proses compile terhadap mode. Kita akan melakukan Insialisasi fungsi, membuat layer embbeding user, layer embedding user bias, dan layer embeddings resto

**Mendapatkan Rekomendasi Pariwisata**
Untuk mendapatkan rekomendasi pariwisata, pertama kita ambil sampel user secara acak dan definisikan variabel tourism_not_visited yang merupakan daftar pariwisata yang belum pernah dikunjungi oleh pengguna. Anda mungkin bertanya-tanya, mengapa kita perlu menentukan daftar tourism_not_visited? Hal ini karena daftar tourism_not_visited inilah yang akan menjadi pariwisata yang kita rekomendasikan. 
Sebelumnya, pengguna telah memberi rating pada beberapa pariwisata yang telah mereka kunjungi. Kita menggunakan rating ini untuk membuat rekomendasi pariwisata yang mungkin cocok untuk pengguna. Nah, pariwisata yang akan direkomendasikan tentulah pariwisata yang belum pernah dikunjungi oleh pengguna. Oleh karena itu, kita perlu membuat variabel tourism_not_visited sebagai daftar pariwisata untuk direkomendasikan pada pengguna. 
Variabel tourism_not_visited diperoleh dengan menggunakan operator bitwise (~) pada variabel tourism_visited_by_user.

Untuk memperoleh rekomendasi pariwisata, gunakan fungsi model.predict() dari library keras. Hasil Rekomendasi untuk user dengan id 154 dapat kita lihat pada Tabel 3.

Tabel 3. Top 10 turism recommendation 
||Tourism with high ratings from user|Top 10 tourism recommendation|
|---|---|---|
|1|Waterboom PIK (Pantai Indah Kapuk) : Taman Hiburan|Skyrink - Mall Taman Anggrek : Taman Hiburan|
|2|Pantai Baron : Bahari|Bumi Perkemahan Cibubur : Taman Hiburan|
|3|Green Village Gedangsari : Taman Hiburan|Galeri Nasional Indonesia : Budaya|
|4|Pantai Congot : Bahari|Monumen Selamat Datang : Budaya|
|5|Gunung Lalakon : Cagar Alam|Taman Situ Lembang : Taman Hiburan|
|6||Jakarta Planetarium : Taman Hiburan|
|7||Kampung Cina : Budaya|
|8||Rumah Sipitung : Budaya|
|9||Museum Tekstil : Budaya|
|10||Bukit Moko : Cagar Alam|
Dari Tabel 3 tersebut, kita dapat membandingkan antara pariwisata dengan rating tinggi dari pengguna dan rekomendasi 10 pariwisata teratas.
Perhatikanlah, beberapa pariwisata rekomendasi menyediakan kategori objek wisata yang sesuai dengan rating pengguna.kita memperoleh 4 rekomendasi pariwisata dengan kategori Taman Hiburan, dan 1 pariwisata dengan kategori Cagar Alam.
 kita telah berhasil membuat sistem rekomendasi dengan teknik Collaborative Filtering. Sistem rekomendasi yang kita buat telah berhasil memberikan sejumlah rekomendasi pariwisata yang sesuai dengan preferensi pengguna. 
 
### Evaluation
Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam(Adaptive Moment Estimation) sebagai optimizer dengan learning_rate=0.001, dan root mean squeared error(RMSE) sebagai metrics evaluation.

**Visualisasi Metrik**
Untuk melihat visualisasi proses training, mari kita plot metrik evaluasi dengan matplotlib
[![Screenshot-2022-09-21-at-11-00-48-Google-Colaboratory.png](https://i.postimg.cc/MKpVZKgJ/Screenshot-2022-09-21-at-11-00-48-Google-Colaboratory.png)](https://postimg.cc/G96HqrcX)


Dari proses ini, kita memperoleh nilai error pada data training akhir sebesar sekitar 0.315 dan error pada data validasi sebesar 0.36. Nilai tersebut cukup bagus untuk sistem rekomendasi.