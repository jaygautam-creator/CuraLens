"""
CuraLens v2 - Multi-Modal Cancer Detection Model
=================================================
Architecture:
  - Branch 1: CNN (EfficientNetB0) for image feature extraction
  - Branch 2: Dense network for patient metadata
  - Fusion: Concatenation of both feature vectors
  - Head: Fully-connected layers → Sigmoid binary classification

Metadata inputs:
  age           : float  (normalised 0-1 externally, or pass raw for BatchNorm)
  smoking       : float  0 or 1
  alcohol       : float  0 or 1
  sun_exposure  : float  0 or 1

NOTE: This module is independent of the existing v1 model. Do NOT import
      or modify any file under models/ or modules/.
"""

import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.applications import EfficientNetB0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)          # EfficientNetB0 native input
IMAGE_SHAPE = IMAGE_SIZE + (3,)  # (H, W, C)
METADATA_DIM = 4                 # age, smoking, alcohol, sun_exposure


# ---------------------------------------------------------------------------
# Sub-network builders
# ---------------------------------------------------------------------------

def build_image_branch(image_input: tf.Tensor,
                       trainable_base: bool = False) -> tf.Tensor:
    """
    CNN branch using EfficientNetB0 as a feature extractor.

    Args:
        image_input:    Keras symbolic tensor of shape IMAGE_SHAPE.
        trainable_base: Whether to fine-tune the EfficientNet weights.

    Returns:
        Feature tensor of shape (batch, 1280) before fusion.
    """
    # Load pre-trained EfficientNetB0; exclude the top classifier
    base_cnn = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
        pooling="avg",          # Global Average Pooling → (batch, 1280)
    )
    base_cnn.trainable = trainable_base

    # Optional lightweight projection head to regularise the image embedding
    x = base_cnn.output
    x = layers.Dense(512, activation="relu", name="img_proj")(x)
    x = layers.Dropout(0.4, name="img_dropout")(x)
    return x                   # (batch, 512)


def build_metadata_branch(metadata_input: tf.Tensor) -> tf.Tensor:
    """
    Dense branch for structured patient metadata.

    Args:
        metadata_input: Keras symbolic tensor of shape (METADATA_DIM,).

    Returns:
        Feature tensor of shape (batch, 64) before fusion.
    """
    x = layers.BatchNormalization(name="meta_bn")(metadata_input)
    x = layers.Dense(64, activation="relu", name="meta_dense_1")(x)
    x = layers.Dropout(0.3, name="meta_dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="meta_dense_2")(x)
    x = layers.Dropout(0.3, name="meta_dropout_2")(x)
    return x                   # (batch, 64)


# ---------------------------------------------------------------------------
# Full multi-modal model
# ---------------------------------------------------------------------------

def build_multimodal_model(trainable_cnn: bool = False,
                           learning_rate: float = 1e-4) -> Model:
    """
    Assemble the full multi-modal model and compile it.

    Args:
        trainable_cnn:  Fine-tune EfficientNet weights (default False for
                        transfer-learning warm-up phase).
        learning_rate:  Adam learning rate.

    Returns:
        Compiled Keras Model with two inputs:
            - "image_input"    : (batch, 224, 224, 3)
            - "metadata_input" : (batch, 4)
        and one output:
            - "cancer_prob"    : (batch, 1)  sigmoid probability
    """
    # ---- Inputs ----------------------------------------------------------------
    image_input    = Input(shape=IMAGE_SHAPE,    name="image_input")
    metadata_input = Input(shape=(METADATA_DIM,), name="metadata_input")

    # ---- Branches --------------------------------------------------------------
    img_features  = build_image_branch(image_input, trainable_base=trainable_cnn)
    meta_features = build_metadata_branch(metadata_input)

    # ---- Fusion ----------------------------------------------------------------
    fused = layers.Concatenate(name="fusion")([img_features, meta_features])
    # (batch, 576)

    # ---- Classification head ---------------------------------------------------
    x = layers.Dense(256, activation="relu", name="head_dense_1")(fused)
    x = layers.Dropout(0.5, name="head_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense_2")(x)
    x = layers.Dropout(0.3, name="head_dropout_2")(x)
    output = layers.Dense(1, activation="sigmoid", name="cancer_prob")(x)

    # ---- Assemble --------------------------------------------------------------
    model = Model(
        inputs=[image_input, metadata_input],
        outputs=output,
        name="CuraLens_v2_MultiModal",
    )

    # ---- Compile ---------------------------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_model(model: Model, path: str) -> None:
    """Save the model in TensorFlow SavedModel format."""
    model.save(path)
    print(f"[CuraLens v2] Model saved → {path}")


from utils_v2.model_loader import safe_load_model

def load_model(path: str) -> Model:
    """Load a previously saved v2 model from disk."""
    model = safe_load_model(path)
    print(f"[CuraLens v2] Model loaded ← {path}")
    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    model = build_multimodal_model()
    model.summary()

    # Dummy forward pass
    dummy_images   = np.random.rand(2, 224, 224, 3).astype("float32")
    dummy_metadata = np.array([[35, 1, 0, 1], [60, 0, 1, 0]], dtype="float32")
    predictions    = model.predict([dummy_images, dummy_metadata])
    print("Dummy predictions:", predictions)
