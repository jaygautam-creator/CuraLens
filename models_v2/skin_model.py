"""
CuraLens v2 — Skin Cancer Multimodal Model
===========================================
EfficientNetB0 image branch + 6-feature clinical skin metadata branch.

Metadata inputs (6D — "skin" schema from utils_v2/metadata_schema.py):
  age                    : float (z-score normalised)
  skin_type              : int   Fitzpatrick 1–6 (z-score normalised)
  sunburn_history        : int   (log1p + z-score)
  outdoor_hours_per_week : float (z-score normalised)
  tanning_bed_use        : bool  (0 / 1)
  family_history         : bool  (0 / 1)

Save/load path:
  models_v2/skin_saved_model/   ← primary

Training data expected under:
  skin_dataset_resized/train_set/{benign,malignant}/
  skin_dataset_resized/val_set/{benign,malignant}/
  skin_dataset_resized/test_set/{benign,malignant}/

NOTE: This module is independent of oral_model.py, multimodal_model.py,
      modules/skin_screening.py and all v1 code.
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
IMAGE_SHAPE  = IMAGE_SIZE + (3,)
METADATA_DIM = 6                           # 6-feature clinical skin schema

_MODULE_DIR      = Path(__file__).resolve().parent
SKIN_SAVED_MODEL = _MODULE_DIR / "skin_saved_model"
SKIN_CHECKPOINT  = _MODULE_DIR / "best_skin_ckpt.h5"


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
    x = layers.Dense(512, activation="relu", name="skin_img_proj")(x)
    x = layers.Dropout(0.4, name="skin_img_dropout")(x)
    return x


def _metadata_branch(meta_input: tf.Tensor) -> tf.Tensor:
    """Dense network for 6-feature skin metadata → (batch, 64)."""
    x = layers.BatchNormalization(name="skin_meta_bn")(meta_input)
    x = layers.Dense(64, activation="relu", name="skin_meta_d1")(x)
    x = layers.Dropout(0.3, name="skin_meta_drop1")(x)
    x = layers.Dense(64, activation="relu", name="skin_meta_d2")(x)
    x = layers.Dropout(0.3, name="skin_meta_drop2")(x)
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_skin_model(
    trainable_cnn: bool  = False,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Assemble and compile the CuraLens Skin multimodal model.

    Args:
        trainable_cnn : Whether to fine-tune EfficientNet weights.
        learning_rate : Adam learning rate.

    Returns:
        Compiled Keras Model with two inputs:
            "skin_image_input"    : (batch, 224, 224, 3)
            "skin_metadata_input" : (batch, 6)
        One output:
            "skin_cancer_prob"    : (batch, 1) sigmoid
    """
    img_input  = Input(shape=IMAGE_SHAPE,     name="skin_image_input")
    meta_input = Input(shape=(METADATA_DIM,), name="skin_metadata_input")

    img_feats  = _image_branch(img_input, trainable_base=trainable_cnn)
    meta_feats = _metadata_branch(meta_input)

    fused = layers.Concatenate(name="skin_fusion")([img_feats, meta_feats])

    x = layers.Dense(256, activation="relu", name="skin_head_d1")(fused)
    x = layers.Dropout(0.5, name="skin_head_drop1")(x)
    x = layers.Dense(128, activation="relu", name="skin_head_d2")(x)
    x = layers.Dropout(0.3, name="skin_head_drop2")(x)
    output = layers.Dense(1, activation="sigmoid", name="skin_cancer_prob")(x)

    model = Model(
        inputs  = [img_input, meta_input],
        outputs = output,
        name    = "CuraLens_Skin_MultiModal",
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

def save_skin_model(model: Model, path: str | Path | None = None) -> None:
    """Save the skin model in TensorFlow SavedModel format."""
    save_to = str(path or SKIN_SAVED_MODEL)
    model.save(save_to)
    print(f"[CuraLens Skin] Model saved → {save_to}")


def load_skin_model(path: str | Path | None = None) -> Model:
    """
    Load the skin multimodal model from disk.

    Resolution order:
      1. `path` if provided and exists
      2. models_v2/skin_saved_model/
      3. Build fresh random-weight model (with warning)
    """
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(SKIN_SAVED_MODEL)

    for candidate in candidates:
        # A SavedModel directory must contain saved_model.pb to be valid.
        if candidate.exists():
            print(f"[CuraLens Skin] Loading model from: {candidate}")
            return safe_load_model(str(candidate))

    raise FileNotFoundError(
        f"Skin V2 model not found at {SKIN_SAVED_MODEL}. "
        "Run: python train_v2.py --cancer-type skin"
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    model = build_skin_model()
    model.summary()

    dummy_imgs = np.random.rand(2, 224, 224, 3).astype("float32")
    dummy_meta = np.random.rand(2, 6).astype("float32")
    preds = model.predict([dummy_imgs, dummy_meta], verbose=0)
    print("Skin dummy predictions:", preds.flatten())
