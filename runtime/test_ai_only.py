import numpy as np
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "admosys_model.keras")
MEAN_PATH = os.path.join(BASE_DIR, "model", "X_mean.npy")
STD_PATH = os.path.join(BASE_DIR, "model", "X_std.npy")

model = tf.keras.models.load_model(MODEL_PATH)
X_mean = np.load(MEAN_PATH)
X_std = np.load(STD_PATH)

# Simulated feature vector
features = np.random.rand(9)

features_norm = (features - X_mean) / X_std
prediction = model.predict(features_norm.reshape(1, -1), verbose=0)

print("AI Output:", prediction)