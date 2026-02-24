import numpy as np
import tensorflow as tf

# load INT8 TFLite model
interpreter = tf.lite.Interpreter(
    model_path="export/admosys_model_int8.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# example feature vector (9 features)
features = np.array([
    1200,   # avg_rpm
    20,     # rpm_slope
    0.6,    # avg_torque
    0.1,    # torque_var
    0.02,   # current_ripple
    0.025,  # vib_rms
    1.7,    # vib_peak_ratio
    60,     # motor_temp
    0.4     # temp_slope
], dtype=np.float32)

# quantize input
scale, zero_point = input_details[0]['quantization']
features_int8 = (features / scale + zero_point).astype(np.int8)

interpreter.set_tensor(
    input_details[0]['index'],
    features_int8.reshape(1, -1)
)

interpreter.invoke()

# get output
output_int8 = interpreter.get_tensor(output_details[0]['index'])

# dequantize output
out_scale, out_zero = output_details[0]['quantization']
output = (output_int8.astype(np.float32) - out_zero) * out_scale

print("INT8 AI output:", output)
