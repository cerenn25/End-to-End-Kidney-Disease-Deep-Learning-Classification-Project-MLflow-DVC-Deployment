# End-to-End-Kidney-Disease-Deep-Learning-Classification-Project-MLflow-DVC-Deployment


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py

1. Genel Durum Analizi: Bir Önceki Modele Göre Muazzam Gelişme
• "Daha önce modelimiz %81 eğitim, %62 test başarısıyla ciddi bir Overfitting (Ezberleme) problemine sahipti. Yapısal bir iyileştirmeyle modele L2 Regularization ve %50 Dropout ekledik. Bu müdahale sonucunda modelimizin genel genelleme (generalization) yeteneği arttı. Epoch 1'de test başarımız (val_accuracy) %62'den %82.14'e fırladı ve ezberleme eğilimi büyük ölçüde kırıldı."

2. Kritik Hata Analizi: Epoch 2 ve Epoch 4'teki Veri Tıkanıklığı (Data Leakage/Pipeline Bug)
Eğitim veri akış hattımızda (Data Pipeline / Generator) yapısal bir problem var. Görüntü jeneratörümüz 2. ve 4. turlarda (Epoch) tüm veriyi dönmek yerine sadece 1 batch (1/459) veri döndükten sonra veri bitti uyarısı verip o turu erken kapatmış. Epoch 2'deki %87.50 doğruluk oranı yanıltıcıdır; çünkü model o turda 7350 görsel yerine sadece 32 görsel görüp turu bitirmiştir. Pipeline'daki steps_per_epoch hesabı veya generator'ın kendini tekrarlama (.repeat()) mekanizması hatalıdır, düzeltilmesi gerekir.

3. Trend Analizi: Model Kararsızlığı (Fluctuation) ve Optimizasyon Sorunu
Model stabil bir iyileşme trendine sahip değil. Epoch 3'te eğitim doğruluğu %88, test doğruluğu %83 iken; veri tıkanıklığı yaşayan Epoch 4'ten hemen sonra Epoch 5'te eğitim başarısı %90.37'ye çıkmasına rağmen test başarısı %74.45'e gerilemiş. Son turda val_loss değerinin 3.22'den 7.08'e fırlaması modelin o turda tekrar hafifçe ezberlemeye (overfitting) başladığına işaret ediyor.

1. Data Pipeline'ı Sabitlemek: "İlk iş olarak Keras'ın veri tüketim uyarısını gidermek için veri yükleme (dataflow_kwargs) fonksiyonundaki steps_per_epoch parametresinin hesaplamasını (toplam_veri / batch_size) gözden geçireceğim."
2. Öğrenme Oranını (Learning Rate) Düşürmek: "Model turlar arasında dalgalı (kararsız) hareket ediyor. params.yaml dosyasındaki LEARNING_RATE (öğrenme oranı) değerini bir tık düşürerek (Örn: 0.01'den 0.001'e) modelin daha yumuşak ve kararlı öğrenmesini sağlayacağım."
3. Erken Durdurma (Early Stopping) Eklemek: "Modelin en iyi performans gösterdiği an Epoch 3 (val_loss: 3.22, val_accuracy: %83.24) anıydı. Epoch 5'e kadar gidip modeli yormak ve ezberletmek yerine, Keras EarlyStopping callback'i ekleyerek eğitimi en optimum noktada otomatik olarak sonlandıracağım."

Böbrek CT taramalarını sınıflandırmak için geliştirdiğim uçtan uca derin öğrenme projesini tamamladım! Proje sırasında Keras 3 ile gelen yenilikleri ve Windows ortamındaki güvenlik duvarlarının (Smart App Control - WinError 4551) veri hattını nasıl etkilediğini deneyimleme şansım oldu. Overfitting sorununu çözmek için L2 Regularization ve Early Stopping mekanizmalarını entegre ettim, deneyleri MLflow ile takip ettim. Detaylar ve kaynak kodlar GitHub depomda...

Ben bu projeyi yaparken sadece hazır kodu kopyalayıp çalıştırmadım. Keras 3 geçişindeki RecursionError mimari bug'ını, Windows altındaki WinError 4551 terminal yetki kısıtlamalarını bizzat debug ederek çözdüm. Modeli ezberden kurtarmak için üretim standartlarında Early Stopping ve L2 entegrasyonu kurdum.

Epoch sayısını güvenle 15 veya 20'ye çıkaracağım ancak yukarıdaki dezavantajlardan (zaman kaybı ve overfitting) kaçınmak için eğitime daha önce konuştuğumuz Early Stopping ve ReduceLROnPlateau (Gelişme durduğunda öğrenme hızını düşüren) callback mekanizmalarını dahil edeceğim. Böylece model eğer 7. epoch'ta en iyi haline ulaşır ve ezberlemeye başlarsa, 20. epoch'a kadar boşuna dönmeyecek ve eğitimi otomatik olarak keserek bilgisayarı kapatacaktır.
Modeli canlıya alırken maliyet optimizasyonu adına mimariyi AWS Lambda gibi Serverless bir yapı üzerine kurmayı tercih ederim. VGG16 büyük bir model olduğu için 'Cold Start' süresini düşürmek amacıyla modeli canlıya çıkmadan önce ONNX formatına dönüştürüp optimize ederim. Ayrıca sağlık verisi işlediğimiz için KVKK uyumluluğuna ve API Gateway arkasında veri güvenliğine dikkat ederim.
Modelimi internete doğrudan açmak yerine, AWS API Gateway arkasındaki bir Private VPC (Özel Sanal Ağ) içine konumlandırırım. Gateway üzerinde OAuth 2.0 ile kimlik doğrulaması yapar ve Throttling kuralları ile DDoS saldırılarını engellerim. KVKK/HIPAA uyumluluğu için veri trafiğini uçtan uca HTTPS ile şifreler, medikal görsellerdeki (DICOM) hasta kimlik bilgilerini modele ulaşmadan önce Gateway katmanında temizleyerek veri minimizasyonu sağlarım.
video
Yukarıdaki demoda, eğitilmiş modern .keras (VGG16 + L2 Regularization + Dropout) modelinin lokal ortamda FastAPI sunucusu üzerinden nasıl çalıştığı gösterilmektedir. Geliştirilen /predict endpoint'ine gönderilen ham böbrek CT tarama görüntüsü anlık olarak işlenmekte ve milisaniyeler içinde sınıflandırma sonucu (Kist/Tümör/Normal) dönmektedir.

# How to run?

### STEPS:

Clone the repository

```bash
git clone https://github.com/cerenn25/End-to-End-Kidney-Disease-Deep-Learning-Classification-Project-MLflow-DVC-Deployment.git
```
### STEP 01- Create a conda environment after opening the repository 

```bash
conda create -n kidney python=3.12 -y
```
```bash
conda activate kidney
```

### STEP 02- Install the requirements

```bash
pip install -r requirements.txt
```



