
import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow.keras.models import load_model
from datetime import datetime

# Load trained model and class labels
model = load_model("models/hand_gesture_model.h5")
class_names = sorted(os.listdir("data/asl_alphabet_train"))  # Get correct labels

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Confidence threshold
confidence_threshold = 0.7

# Output directory
base_save_dir = "captured_predictions"
os.makedirs(base_save_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Make sure the landmark input matches the model's expected shape
            if len(landmarks) == model.input_shape[1]:
                input_data = np.expand_dims(landmarks, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]
                max_prob = np.max(prediction)

                if max_prob >= confidence_threshold:
                    pred_idx = np.argmax(prediction)
                    predicted_label = class_names[pred_idx]

                    # Annotate the frame
                    cv2.putText(frame, f"Pred: {predicted_label}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Save frame in label-specific folder
                    label_dir = os.path.join(base_save_dir, predicted_label)
                    os.makedirs(label_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    filename = f"{predicted_label}_{timestamp}.jpg"
                    save_path = os.path.join(label_dir, filename)
                    cv2.imwrite(save_path, frame)
                    print(f"Saved: {save_path}")

            # Draw landmarks on frame
            mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow("Sign Language Live Capture", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
