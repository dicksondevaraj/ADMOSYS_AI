import numpy as np
import tensorflow as tf
try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None
    print("Running in simulation mode (No GPIO)")
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "admosys_model.keras")
MEAN_PATH = os.path.join(BASE_DIR, "model", "X_mean.npy")
STD_PATH = os.path.join(BASE_DIR, "model", "X_std.npy")

# ===============================
# -------- LOAD AI MODEL --------
# ===============================

model = tf.keras.models.load_model(MODEL_PATH)
X_mean = np.load(MEAN_PATH)
X_std = np.load(STD_PATH)

# ===============================
# -------- CONFIGURATION --------
# ===============================

STROKE_LENGTH = 100.0        # mm
SPEED_FULL_LOAD = 4.2        # mm/sec
UPDATE_INTERVAL = 0.5        # seconds (slow mechanical loop)
MOVE_DURATION = 0.3          # seconds per movement
POSITION_TOLERANCE = 3.0     # mm
COOLDOWN = 2.0               # seconds between movements
PWM_DUTY = 60                # % (safe limit)



# ===============================
# GPIO SETUP
# ===============================

# GPIO setup
IN1 = 23
IN2 = 24
PWM_PIN = 18

if GPIO:
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(PWM_PIN, GPIO.OUT)

    pwm = GPIO.PWM(PWM_PIN, 1000)
    pwm.start(0)

    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)

else:
    print("Running in simulation mode (No GPIO)")
    pwm = None



# ===============================
# ---- POSITION ESTIMATOR -------
# ===============================

current_position_mm = 50.0  # start at mid stroke
last_move_time = 0


def update_position(direction, duration):
    global current_position_mm

    distance = SPEED_FULL_LOAD * duration

    if direction == "extend":
        current_position_mm += distance
    elif direction == "retract":
        current_position_mm -= distance

    current_position_mm = max(0, min(STROKE_LENGTH, current_position_mm))


# ===============================
# ----- MOTOR CONTROL -----------
# ===============================

def stop_motor():
    if GPIO and pwm:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        pwm.ChangeDutyCycle(0)


def extend(duration):
    if GPIO and pwm:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        pwm.ChangeDutyCycle(PWM_DUTY)
        time.sleep(duration)
        stop_motor()
    else:
        print("Simulated extend")


def retract(duration):
    if GPIO and pwm:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        pwm.ChangeDutyCycle(PWM_DUTY)
        time.sleep(duration)
        stop_motor()
    else:
        print("Simulated retract")


# ===============================
# ---- FEATURE EXTRACTION -------
# ===============================

dt = 0.02

def extract_features(rpm, torque, current, vibration, temperature):
    avg_rpm = np.mean(rpm)
    rpm_slope = (rpm[-1] - rpm[0]) / dt

    avg_torque = np.mean(torque)
    torque_var = np.var(torque)

    current_ripple = np.sqrt(np.mean((current - np.mean(current))**2))

    vib_rms = np.sqrt(np.mean(vibration**2))
    vib_peak_ratio = np.max(vibration) / (vib_rms + 1e-6)

    motor_temp = temperature[-1]
    temp_slope = (temperature[-1] - temperature[0]) / dt

    return np.array([
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


# ===============================
# -------- AI FUNCTION ----------
# ===============================

def run_ai(features):
    features_norm = (features - X_mean) / X_std
    prediction = model.predict(features_norm.reshape(1, -1), verbose=0)
    return prediction[0][0]   # use first output for stiffness


def ai_to_target_mm(ai_value):
    stiffness = (ai_value + 1) / 2   # convert -1 to +1 → 0 to 1
    return stiffness * STROKE_LENGTH


# ===============================
# -------- MAIN LOOP ------------
# ===============================
prev_ai = 0
alpha = 0.3   # smoothing factor

try:
    while True:

        # -------- Replace with real sensor readings --------
        rpm = np.random.uniform(1000, 1500, 5)
        torque = np.random.uniform(0.4, 0.7, 5)
        current = np.random.uniform(10, 15, 5)
        vibration = np.random.uniform(0.01, 0.04, 5)
        temperature = np.random.uniform(50, 65, 5)
        # ---------------------------------------------------

        features = extract_features(rpm, torque, current, vibration, temperature)

        ai_value = np.sin(time.time())   # run_ai(features)  # Uncomment to use AI prediction

        # Smooth AI output
        ai_value = alpha * ai_value + (1 - alpha) * prev_ai
        prev_ai = ai_value

        # Clamp safety
        ai_value = np.clip(ai_value, -1, 1)

        target_mm = ai_to_target_mm(ai_value)
        error = target_mm - current_position_mm

        print(f"Target: {target_mm:.2f} mm | Current: {current_position_mm:.2f} mm")

        if abs(error) > POSITION_TOLERANCE:
            if time.time() - last_move_time > COOLDOWN:

                if error > 0 and current_position_mm < STROKE_LENGTH:
                    extend(MOVE_DURATION)
                    update_position("extend", MOVE_DURATION)

                elif error < 0 and current_position_mm > 0:
                    retract(MOVE_DURATION)
                    update_position("retract", MOVE_DURATION)

                last_move_time = time.time()

        time.sleep(UPDATE_INTERVAL)

except KeyboardInterrupt:
    stop_motor()
    if GPIO:
        GPIO.cleanup() 