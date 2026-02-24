import tensorflow as tf

model_path = "export/admosys_model_int8.tflite"

analysis = tf.lite.experimental.Analyzer.analyze(
    model_path=model_path
)

print(analysis)
