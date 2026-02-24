import numpy as np
import tensorflow as tf
# simulate 20 ms window (10 samples)
rpm = np.array([1180, 1190, 1200, 1210, 1220])
torque = np.array([0.55, 0.58, 0.60, 0.63, 0.65])
current = np.array([12.1, 12.3, 12.2, 12.4, 12.6])
vibration = np.array([0.02, 0.021, 0.022, 0.024, 0.026])
temperature = np.array([54.8, 54.9, 55.0, 55.1, 55.2])

dt = 0.02  # 20 ms
def extract_features(rpm, torque, current, vibration, temperature, dt):
    features = []

    # speed behavior
    avg_rpm = np.mean(rpm)
    rpm_slope = (rpm[-1] - rpm[0]) / dt

    # load behavior
    avg_torque = np.mean(torque)
    torque_var = np.var(torque)

    # electrical stress
    current_ripple = np.sqrt(np.mean((current - np.mean(current))**2))

    # mechanical stress
    vib_rms = np.sqrt(np.mean(vibration**2))
    vib_peak_ratio = np.max(vibration) / (vib_rms + 1e-6)

    # thermal state
    motor_temp = temperature[-1]
    temp_slope = (temperature[-1] - temperature[0]) / dt

    features.extend([
        avg_rpm,
        rpm_slope,
        avg_torque,
        torque_var,
        current_ripple,
        vib_rms,
        vib_peak_ratio,
        motor_temp,
        temp_slope
    ])

    return np.array(features)
features = extract_features(
    rpm, torque, current, vibration, temperature, dt
)

print("Feature vector:", features)
print("Feature count:", len(features))
# fake dataset (we'll replace later)
X = []
Y = []

for _ in range(200):
    rpm = np.random.uniform(800, 2000, 5)
    torque = np.random.uniform(0.3, 0.9, 5)
    current = np.random.uniform(8, 18, 5)
    vibration = np.random.uniform(0.01, 0.05, 5)
    temperature = np.random.uniform(40, 80, 5)

    features = extract_features(
        rpm, torque, current, vibration, temperature, dt
    )

    # simple logic-based labels (engineering intuition)
    delta_kp = -0.05 if features[5] > 0.03 else 0.02
    delta_iq = -0.08 if features[7] > 70 else 0.01

    X.append(features)
    Y.append([delta_kp, delta_iq])

X = np.array(X)
Y = np.array(Y)

# --- Normalize inputs ---
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0) + 1e-6

# Save normalization parameters
np.save("model/X_mean.npy", X_mean)
np.save("model/X_std.npy", X_std)

X = (X - X_mean) / X_std

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(9,)),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(2, activation="tanh")
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()
model.fit(X, Y, epochs=50, batch_size=16, verbose=1)
features_norm = (features - X_mean) / X_std
test_output = model.predict(features_norm.reshape(1, -1))
print("AI suggestion:", test_output)
# save trained model
model.save("model/admosys_model.keras")
print("Model saved")
