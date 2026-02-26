import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "admosys_model.keras")
EXPORT_PATH = os.path.join(BASE_DIR, "export", "admosys_model.tflite")

# Load Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save file
with open(EXPORT_PATH, "wb") as f:
    f.write(tflite_model)

print("TFLite model exported successfully.")