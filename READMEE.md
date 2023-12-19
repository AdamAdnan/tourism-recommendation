# Laporan Proyek Machine Learning - Adam Adna

### Project Overview
Meskipun Indonesia memiliki tempat wisata yang menarik - pedalaman yang indah, reruntuhan budaya dan sejarah yang menarik, pantai, dan banyak lagi. Namun masih banyak turis asing maupun lokal yang tidak mengetahui lokasi tempat wisata tersebut.
Masalahnya turis asing maupun lokal akan merasa bosan jika hanya mengetahui satu tempat wisata saja. Akibatnya, Jumlah yang berkunjung akan berkurang.

Dari masalah di atas Bayangkan kita adalah seorang Data Scientist di perusahaan pariwisata. Setelah sekian lama beroperasi, perusahaan kita berhasil mengumpulkan berbagi informasi mengenai pengguna, pariwisata dan objek wisata, serta rating yang diberikan pengguna untuk pariwisata tersebut. Seluruh informasi ini terkumpul dalam Tourism and destination.

### Business Understanding
Sebagai seorang Data Scientis, untuk kita ingin memanfaatkan data tersebut untuk meningkatkan transaksi di perusahaan. Kembangkan sebuah sistem rekomendasi pariwisata untuk menjawab permasalahan tersebut:
- Dengan data rating yang Anda miliki, bagaimana perusahaan dapat merekomendasikan tujuan wisata lain yang mungkin disukai dan belum pernah dikunjungi oleh pengguna?
- Menghasilkan sejumlah rekomendasi tujuan wisata yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi sebelumnya dengan teknik collaborative filtering.

### Data Understanding 
Dataset yang digunakan pada proyek ini adalah Indonesia Tourism Destination yang dapat diunduh dari https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/code . Kita akan menggunakan Google Colaboratory untuk membuat model sistem rekomendasi.
Pertama, Kita download dulu dataset. Jika berhasil, file indonesia-tourism-destination.zip akan masuk ke sistem storage Google Colab.
