
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import os

# Load landmark data
X = np.load("data/X_landmarks.npy")
y = np.load("data/y_labels.npy")

# Create class names from your asl_alphabet_train folder
class_names = sorted(os.listdir("data/asl_alphabet_train"))

# Make sure labels are categorical
y = to_categorical(y, num_classes=len(class_names))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the improved model
model = Sequential([
    Dense(256, input_shape=(X.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(len(class_names), activation='softmax')
])

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/hand_gesture_model.h5")
print("✅ Model saved at models/hand_gesture_model.h5")
