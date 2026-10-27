import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def prediction(self):
        # 1. Modeli yükle
        model = load_model(os.path.join("artifacts", "training", "model.keras"))

        # 2. Resmi yükle ve boyutlandır
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        
        # 3. NORMALİZASYON (Kritik Adım: Pikselleri 0-1 arasına sıkıştırıyoruz)
        test_image = test_image / 255.0
        
        # 4. Batch boyutu ekle (1, 224, 224, 3)
        test_image = np.expand_dims(test_image, axis=0)
        
        # 5. Modelden tahmini al ve terminale yazdır
        predictions = model.predict(test_image)
        print("Modelin Ham Çiktisi (Olasiliklar):", predictions)
        
        # 6. İki sınıflı yapılarda güvenli indeks bulma
        # Eğer model tek bir çıktı üretiyorsa (Binary/Sigmoid)
        if predictions.shape[-1] == 1:
            predicted_index = 1 if predictions[0][0] > 0.5 else 0
        # Eğer model iki çıktı üretiyorsa (Categorical/Softmax)
        else:
            predicted_index = np.argmax(predictions, axis=1)[0]
            
        print("Tahmin Edilen Sinif İndeksi:", predicted_index)

        # 7. Sınıf Eşleştirmesi (Alfabetik olarak Normal, Tumor'den önce gelir)
        # Klasör yapınız: 0 -> Normal, 1 -> Tumor şeklindedir
        classes = ["Normal", "Tumor"]
        prediction = classes[predicted_index]

        return [{"image" : prediction}]
