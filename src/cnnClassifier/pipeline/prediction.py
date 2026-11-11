""" import os
import numpy as np
from PIL import Image # kerasın orijinal görsel yükleyicisi
import onnxruntime as ort

class PredictionPipeline:
    def __init__(self,filename=None):
        self.filename = filename

        # ONNX Modelini yükle
        model_path = os.path.join("model", "model.onnx")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def prediction(self):
        imagename = self.filename if self.filename else "inputImage.jpg"
        
        # 1. Resmi PIL ile açmak
        with Image.open(imagename) as temp_img:
            img = temp_img.convert("RGB").copy()
            
        # 2. Boyutlandırma
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        
        # 3. Numpy array'e çevirme
        test_image = np.array(img, dtype=np.float32)
        
        # -------------------------------------------------------------
        # DENEME ADIMI: Eğer decodeImage resmi BGR kaydettiyse kanalları düzeltiyoruz.
        # Bu satır Mavi ve Kırmızı kanalların yerini değiştirecektir.
        test_image = test_image[:, :, ::-1] 
        # -------------------------------------------------------------
        
        # 4. Normalizasyon (Eğitimdeki gibi 0.0 - 1.0 arası)
        test_image = test_image / 255.0
        
        # 5. Batch boyutu ekle
        test_image = np.expand_dims(test_image, axis=0)
        
        # Squeeze kontrolü (Önceki adımdan kalan emniyet)
        if len(test_image.shape) == 5:
            test_image = np.squeeze(test_image, axis=0)
            
        # 6. ONNX runtime çıkarımı
        raw_outputs = self.session.run(None, {self.input_name: test_image})
        predictions = raw_outputs[0] # Çıktı matrisini doğrudan alın


        print("Modelin Ham Çiktisi (Olasiliklar):", predictions)

        highest_prob = np.max(predictions)

        # 7. Modelin çıktı yapısına göre akllı index bulma
        # Eğer model tek bir olasılık dönüyorsa (Binary- Sigmoid) -> Örn: [[0.95]]
        if predictions.shape[-1] == 1:
            raw_value = predictions[0][0]
            predicted_index = 1 if raw_value >= 0.5 else 0
            highest_prob = raw_value if predicted_index == 1 else (1 - raw_value)
            print(f"Sistem: Binary Algilandi. Değer: {raw_value:.4f}")

        # Eğer model iki olasılık dönüyorsa (Categorical-Softmax) -> Örn: [[0.05, 0.95]]
        else:
            predicted_index = int(np.argmax(predictions, axis=1)[0])
            highest_prob = predictions[0][predicted_index]
            print(f"Sistem: Categorical Algilandi. En yüksek olasilik: {predicted_index}")

        # En yüksek olasılık değerini bulmak
        if highest_prob < 0.70:
            print(f" ⚠️ Model Kararsiz! En yüksek olasilik: {highest_prob: .2f}. Güven eşiği geçilmedi.")
            return [{"image": "Uncertain"}]

        print("Kesinleşen Sinif Indexi:", predicted_index)

        # 9. Sınıf Eşleştirmesi
        classes = ["Normal", "Tumor"]
        prediction = classes[predicted_index]

        return [{"image": prediction}] """

import os
import numpy as np
from PIL import Image
import onnxruntime as ort

class PredictionPipeline:
    def __init__(self, filename=None):
        self.filename = filename
        # ONNX Modelini yükle
        model_path = os.path.join("model", "model.onnx")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def prediction(self):
        # Arayüzden ne gelirse gelsin, geçici olarak elinizle koyduğunuz temiz resmi okutuyoruz:
        imagename = "orijinal_normal.jpg" 
        
        # 1. Resmi PIL ile açmak
        with Image.open(imagename) as temp_img:
            img = temp_img.convert("RGB").copy()
            
        # 2. Boyutlandırma
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        
        # 3. Numpy array'e çevirme
        test_image = np.array(img, dtype=np.float32)
        
        # 4. Kanal düzeltmesi (decodeImage OpenCV kullandığı için BGR->RGB yapıyoruz)
        test_image = test_image[:, :, ::-1] 
        
        # 5. Normalizasyon (0.0 - 1.0 arası)
        test_image = test_image / 255.0
        
        # 6. Batch boyutu ekle (Format: [1, 224, 224, 3] - Modelin tam beklediği şekil)
        test_image = np.expand_dims(test_image, axis=0)
        
        # 7. ONNX runtime çıkarımı
        raw_outputs = self.session.run(None, {self.input_name: test_image})
        
        # KRİTİK NOKTA: ONNX'in liste sarmalından kurtulup ilk çıktı katmanını alıyoruz
        predictions = raw_outputs[0] # Çıktıyı [[0.005, 0.994]] formatına indirgedik
        
        print("Güncel ONNX Model Çıktısı (Olasılıklar):", predictions)
        
        # 8. Sınıflandırma (Her zaman son eksene göre argmax alıyoruz)
        predicted_index = int(np.argmax(predictions, axis=-1)[0])
        
        classes = ["Normal", "Tumor"]
        prediction_result = classes[predicted_index]
        
        return [{"image": prediction_result}]


