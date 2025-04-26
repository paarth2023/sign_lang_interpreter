import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(image, hands_module):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_module.process(image)
    landmarks = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    return None
