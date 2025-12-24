# x-ray-projesi
# MedScan AI - Tıbbi Görüntü Analiz Sistemi

Yapay zeka destekli tıbbi görüntü analiz uygulaması. Akciğer röntgeni, beyin MR ve kemik filmlerini analiz eder.

##  Proje Hakkında

Bu proje, **derin öğrenme** teknikleri kullanarak tıbbi görüntüleri otomatik olarak analiz eden bir web uygulamasıdır. Proje kapsamında üç farklı tıbbi görüntü türü için ayrı ayrı yapay zeka modelleri geliştirilmiştir.

###  Projenin Amacı

Günümüzde sağlık sektöründe yapay zeka giderek daha fazla kullanılmaktadır. Bu projede, tıbbi görüntülerin (röntgen, MR, tomografi) yapay zeka ile nasıl analiz edilebileceğini göstermek amaçlanmıştır. Sistem, doktorlara ön tanı aşamasında yardımcı olabilecek şekilde tasarlanmıştır.

### 🔬 Nasıl Çalışıyor?

1. **Görüntü Yükleme**: Kullanıcı web arayüzünden tıbbi görüntüyü (röntgen, MR vb.) yükler
2. **Ön İşleme**: Görüntü 224x224 piksel boyutuna getirilir, normalize edilir ve model için hazırlanır
3. **Model Tahmini**: Keras ile eğitilmiş CNN (Convolutional Neural Network) modeli görüntüyü analiz eder
4. **Sonuç Gösterimi**: Hastalık sınıfı, güven oranı ve tedavi önerileri kullanıcıya gösterilir

###  Kullanılan Yapay Zeka Teknikleri

Bu projede **Keras** kütüphanesi kullanılarak derin öğrenme modelleri eğitilmiştir:
Transfer Learning yaklaşımı kullanılarak MobileNetV2 modeli üzerine özel katmanlar eklendi ve tıbbi görüntülerle eğitildi.

- **CNN (Convolutional Neural Network)**: Görüntülerden otomatik özellik çıkarımı yapan sinir ağı mimarisi
- **Transfer Learning**: Google tarafından geliştirilen MobileNetV2 modeli temel alınarak, üzerine özel sınıflandırma katmanları eklendi. Bu sayede daha az veriyle yüksek doğruluk elde edildi.
- **Data Augmentation**: Eğitim verisini zenginleştirmek için görüntülere döndürme, yakınlaştırma, yatay çevirme gibi işlemler uygulandı

###  Model Eğitimi

Modeller **Keras** ile şu şekilde eğitildi:


 Özellikler

- Akciğer Analizi: COVID-19, Normal, Pnömoni tespiti
- Beyin Tümör Analizi: Glioma, Meningioma, Pituitary, No Tumor sınıflandırması
- Kemik Kırık Analizi: Kırık var/yok tespiti














