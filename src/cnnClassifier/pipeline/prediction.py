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


