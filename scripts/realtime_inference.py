
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import os

# Load model and classes
model = load_model("models/hand_gesture_model.h5")
class_names = sorted(os.listdir("data/asl_alphabet_train"))  # Match training class order

# Settings
confidence_threshold = 0.7
min_landmarks = 21  # 21 hand landmarks
smoothing_window = 5
pred_queue = deque(maxlen=smoothing_window)

# Init MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        predicted_char = ""

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) >= min_landmarks * 3:
                input_data = np.expand_dims(np.array(landmarks), axis=0)

                prediction = model.predict(input_data, verbose=0)[0]
                max_prob = np.max(prediction)

                if max_prob >= confidence_threshold:
                    pred_idx = np.argmax(prediction)
                    pred_queue.append(pred_idx)

                    # Smoothing: use the mode of recent predictions
                    if len(pred_queue) == smoothing_window:
                        smoothed_idx = max(set(pred_queue), key=pred_queue.count)
                        predicted_char = class_names[smoothed_idx]
                else:
                    pred_queue.clear()

            # Draw landmarks on hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display predicted character
        if predicted_char:
            cv2.putText(frame, f"Prediction: {predicted_char}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Sign Language Interpreter", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
