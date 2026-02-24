import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model/admosys_model.keras")


# representative dataset (VERY IMPORTANT)
def representative_dataset():
    for _ in range(100):
        data = np.random.uniform(-1, 1, (1, 9)).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# enforce INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open("export/admosys_model_int8.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("INT8 TFLite model created")

