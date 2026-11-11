import os
import tf2onnx
import onnx
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Mevcut keras modelini yüklemek
model_path = os.path.join("artifacts", "training", "model.keras")
print(f"Loading Keras model from: {model_path}")
model = load_model(model_path)

# 2. Girdi formatını tamlamak (224, 224, 3)
# Batch boyutunu dinamik olması için None olarak ayarlıyoruz
input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input_layer")]

# 3. ONNX formatına dönüştürmek
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

#4. 'model' adında yeni bir klasör tanımlamak yoksa otomatik oluşturmak
output_dir = "model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"'{output_dir}' klasörü oluşturuldu.")

output_path = os.path.join(output_dir, "model.onnx")
onnx.save(onnx_model, output_path)

print(f"ONNX model saved to: {output_path}")


