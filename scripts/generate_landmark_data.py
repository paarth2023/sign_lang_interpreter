
# scripts/generate_landmark_data.py

import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

mp_hands = mp.solutions.hands

data_dir = "data/asl_alphabet_train"
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

# Read class names properly
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
class_to_index = {class_name: idx for idx, class_name in enumerate(classes)}

X = []
y = []

with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    for label in tqdm(classes, desc="Processing classes"):
        label_path = os.path.join(data_dir, label)
        for img_name in os.listdir(label_path)[:300]:  # Limit to 300 per class
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            if result.multi_hand_landmarks:
                landmarks = []
                for lm in result.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                X.append(landmarks)
                y.append(class_to_index[label])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Save
np.save(os.path.join(save_dir, "X_landmarks.npy"), X)
np.save(os.path.join(save_dir, "y_labels.npy"), y)
np.save(os.path.join(save_dir, "class_names.npy"), np.array(classes))  # <--- NEW

print(f"Saved {len(X)} samples.")
print(f"Classes: {classes}")
