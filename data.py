import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np

# Configuration
IMG_SIZE = 64  # Resize images to 64x64
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = 'sign_data'
MODEL_PATH = 'esrelive_model.h5'

# Gesture classes
CLASSES = ['hello', 'thank_you', 'yes', 'no', 'please', 'goodbye', 'love', 'peace']

print("="*50)
print("🧠 ES RELIVE Model Training")
print("="*50)

# Data augmentation and loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

print("\n📂 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"\n📊 Dataset Summary:")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {validation_generator.samples}")
print(f"   Classes: {list(train_generator.class_indices.keys())}")

# Build CNN Model
def build_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Dense layers
        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

print("\n🏗️ Building model...")
model = build_model()
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

# Train
print("\n🚀 Starting training...")
print("-" * 50)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Save final model
model.save(MODEL_PATH)
print(f"\n✅ Model saved as: {MODEL_PATH}")

# Save class labels
import json
labels = {v: k for k, v in train_generator.class_indices.items()}
with open('class_labels.json', 'w') as f:
    json.dump(labels, f)
print("✅ Class labels saved as: class_labels.json")

# Print results
print("\n" + "="*50)
print("📈 Training Results:")
print("="*50)
print(f"   Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"   Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")