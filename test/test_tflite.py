import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="export/admosys_model_int8.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# example feature vector (same size = 9)
features = np.array([
    0.1, 0.2, 0.3, 0.4
], dtype=np.float32)

# quantize input
scale, zero_point = input_details[0]['quantization']
input_int8 = (features / scale + zero_point).astype(np.int8)

interpreter.set_tensor(input_details[0]['index'], input_int8.reshape(1, -1))
interpreter.invoke()

output_int8 = interpreter.get_tensor(output_details[0]['index'])

# dequantize output
out_scale, out_zero = output_details[0]['quantization']
output = (output_int8.astype(np.float32) - out_zero) * out_scale

print("INT8 AI output:", output)
print("Expected input shape:", input_details[0]['shape'])
