"""
CuraLens v2 — Oral Cancer Multimodal Model
===========================================
EfficientNetB0 image branch + 6-feature clinical metadata branch.

Metadata inputs (6D — "oral" schema from utils_v2/metadata_schema.py):
  age                    : float (z-score normalised)
  smoking_years          : int   (log1p + z-score)
  cigarettes_per_day     : int   (log1p + z-score)
  alcohol_units_per_week : int   (log1p + z-score)
  chewing_tobacco        : bool  (0 / 1)
  family_history         : bool  (0 / 1)

Save/load paths:
  models_v2/oral_saved_model/   ← primary (new 6D trained model)
  models_v2/saved_model/        ← fallback (legacy 4D model still works
                                   for oral_legacy cancer_type requests)

NOTE: This module is architecturally independent from multimodal_model.py
      and the v1 models/. Neither file is imported or modified here.
"""

from __future__ import annotations

from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.applications import EfficientNetB0
from utils_v2.model_loader import safe_load_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE   = (224, 224)
IMAGE_SHAPE  = IMAGE_SIZE + (3,)          # (H, W, C)
METADATA_DIM = 6                           # 6-feature clinical oral schema

# Canonical save directory (relative to this file's parent)
_MODULE_DIR      = Path(__file__).resolve().parent
ORAL_SAVED_MODEL = _MODULE_DIR / "oral_saved_model"
ORAL_CHECKPOINT  = _MODULE_DIR / "best_oral_ckpt.h5"


# ---------------------------------------------------------------------------
# Sub-network builders
# ---------------------------------------------------------------------------

def _image_branch(image_input: tf.Tensor,
                  trainable_base: bool = False) -> tf.Tensor:
    """EfficientNetB0 feature extractor → (batch, 512)."""
    base = EfficientNetB0(
        include_top    = False,
        weights        = "imagenet",
        input_tensor   = image_input,
        pooling        = "avg",
    )
    base.trainable = trainable_base

    x = base.output
    x = layers.Dense(512, activation="relu", name="oral_img_proj")(x)
    x = layers.Dropout(0.4, name="oral_img_dropout")(x)
    return x


def _metadata_branch(meta_input: tf.Tensor) -> tf.Tensor:
    """Dense network for 6-feature oral metadata → (batch, 64)."""
    x = layers.BatchNormalization(name="oral_meta_bn")(meta_input)
    x = layers.Dense(64, activation="relu", name="oral_meta_d1")(x)
    x = layers.Dropout(0.3, name="oral_meta_drop1")(x)
    x = layers.Dense(64, activation="relu", name="oral_meta_d2")(x)
    x = layers.Dropout(0.3, name="oral_meta_drop2")(x)
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_oral_model(
    trainable_cnn: bool  = False,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Assemble and compile the CuraLens Oral v3 multimodal model.

    Args:
        trainable_cnn : Whether to fine-tune EfficientNet weights.
        learning_rate : Adam learning rate.

    Returns:
        Compiled Keras Model with two inputs:
            "oral_image_input"    : (batch, 224, 224, 3)
            "oral_metadata_input" : (batch, 6)
        One output:
            "oral_cancer_prob"    : (batch, 1) sigmoid
    """
    img_input  = Input(shape=IMAGE_SHAPE,     name="oral_image_input")
    meta_input = Input(shape=(METADATA_DIM,), name="oral_metadata_input")

    img_feats  = _image_branch(img_input, trainable_base=trainable_cnn)
    meta_feats = _metadata_branch(meta_input)

    fused = layers.Concatenate(name="oral_fusion")([img_feats, meta_feats])
    # (batch, 576)

    x = layers.Dense(256, activation="relu", name="oral_head_d1")(fused)
    x = layers.Dropout(0.5, name="oral_head_drop1")(x)
    x = layers.Dense(128, activation="relu", name="oral_head_d2")(x)
    x = layers.Dropout(0.3, name="oral_head_drop2")(x)
    output = layers.Dense(1, activation="sigmoid", name="oral_cancer_prob")(x)

    model = Model(
        inputs  = [img_input, meta_input],
        outputs = output,
        name    = "CuraLens_Oral_v3_MultiModal",
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss      = "binary_crossentropy",
        metrics   = [
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_oral_model(model: Model, path: str | Path | None = None) -> None:
    """Save the oral model in TensorFlow SavedModel format."""
    save_to = str(path or ORAL_SAVED_MODEL)
    model.save(save_to)
    print(f"[CuraLens Oral v3] Model saved → {save_to}")


def load_oral_model(
    path: str | Path | None = None,
    fallback_legacy: bool = True,
) -> Model:
    """
    Load the oral model from disk.

    Resolution order:
      1. `path` if provided and exists
      2. models_v2/oral_saved_model/  (new 6D model)
      3. models_v2/saved_model/       (legacy 4D model, if fallback_legacy=True)
      4. Build fresh random-weight model

    Args:
        path            : Explicit path to override resolution.
        fallback_legacy : If True, try the legacy 4D saved_model as last resort.

    Returns:
        Loaded (or freshly built) Keras Model.
    """
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(ORAL_SAVED_MODEL)
    if fallback_legacy:
        candidates.append(_MODULE_DIR / "saved_model")

    for candidate in candidates:
        if candidate.exists():
            print(f"[CuraLens Oral v3] Loading model from: {candidate}")
            return safe_load_model(str(candidate))

    print(
        "[CuraLens Oral v3] WARNING: No saved oral model found. "
        "Building fresh random-weight model. Train before clinical use."
    )
    return build_oral_model(trainable_cnn=False)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    model = build_oral_model()
    model.summary()

    dummy_imgs = np.random.rand(2, 224, 224, 3).astype("float32")
    dummy_meta = np.random.rand(2, 6).astype("float32")
    preds = model.predict([dummy_imgs, dummy_meta], verbose=0)
    print("Oral dummy predictions:", preds.flatten())
