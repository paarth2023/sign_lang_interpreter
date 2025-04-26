
# scripts/test_model.py

import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# Load model and classes
model = load_model("models/hand_gesture_model.h5")
class_names = np.load("models/class_names.npy")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Load test images
test_dir = "data/asl_alphabet_test"
test_images = [img for img in os.listdir(test_dir) if img.endswith(".jpg")]

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks).reshape(1, -1)

        # Predict
        prediction = model.predict(landmarks)
        predicted_class = class_names[np.argmax(prediction)]

        # Draw landmarks and prediction
        mp_drawing.draw_landmarks(img, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.putText(img, f"Predicted: {predicted_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(img, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the image
    cv2.imshow("ASL Test", img)
    key = cv2.waitKey(0)  # Press any key for next image
    if key == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
