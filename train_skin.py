"""
CuraLens – Skin Visual Screening Model Training
Binary classification: Benign vs Malignant
NOTE: AI-assisted screening only (NOT diagnosis)
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

# =====================
# CONFIGURATION
# =====================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

DATASET_DIR = "skin_dataset_resized"
TRAIN_DIR = os.path.join(DATASET_DIR, "train_set")
VAL_DIR = os.path.join(DATASET_DIR, "val_set")

MODEL_DIR = "models/skin_model"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_screening_model.h5")
LOG_PATH = os.path.join(MODEL_DIR, "training_log.csv")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================
# DATA GENERATORS
# =====================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# =====================
# MODEL DEFINITION
# =====================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Transfer learning (safe & stable)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# =====================
# CALLBACKS
# =====================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True),
    CSVLogger(LOG_PATH)
]

# =====================
# TRAINING
# =====================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================
# SAVE METADATA
# =====================
final_auc = max(history.history.get("val_auc", [0]))

metadata = {
    "task": "Skin visual abnormality screening",
    "classes": {
        "0": "Benign (Normal)",
        "1": "Malignant (Abnormal)"
    },
    "architecture": "MobileNetV2 (transfer learning)",
    "input_size": IMAGE_SIZE,
    "performance": {
        "best_val_auc": float(final_auc)
    },
    "ethics": {
        "diagnosis": False,
        "usage": "AI-assisted screening only"
    }
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Skin screening model training completed")
print(f"📁 Model saved to: {MODEL_PATH}")
