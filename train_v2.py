"""
CuraLens v2 — Multi-Modal Training Pipeline
=============================================
Trains the EfficientNetB0 + metadata fusion model in two phases:

  Phase 1 — Warm-up (CNN frozen)
      Train only the fusion head + metadata branch.
      LR = 1e-4, up to 30 epochs, EarlyStopping on val_auc.

  Phase 2 — Fine-tuning (top 20 EfficientNet layers unfrozen)
      Low LR = 1e-5, up to 20 epochs, continue from best Phase-1 weights.

Metadata CSV schema  (data_clean/metadata.csv):
    split          : "train" | "val"
    class_folder   : "zzz_cancer" | "aaa_non_cancer"
    filename       : e.g. "001.jpeg"
    age            : float  (years)
    smoking        : 0 | 1
    alcohol        : 0 | 1
    sun_exposure   : float  (0-10 scale)

If the CSV is absent a synthetic one is generated and saved for reference.
Replace the synthetic rows with real patient records before a clinical study.

Outputs:
    models_v2/saved_model/          ← best model weights (SavedModel format)
    models_v2/training_logs_v2.json ← full epoch-by-epoch history

Run (legacy oral, unchanged):
    python train_v2.py [--epochs-phase1 N] [--epochs-phase2 N] [--batch B]

Run (new skin model):
    python train_v2.py --cancer-type skin [--epochs-phase1 N] [--epochs-phase2 N]

Run (oral with focal loss):
    python train_v2.py --cancer-type oral --use-focal-loss

Run (oral with cross-validation):
    python train_v2.py --cancer-type oral --cross-validate --cv-folds 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
# Force CPU-only training: the Metal (Apple-Silicon) plugin fails with large
# in-memory numpy tensors (~1 GB) in from_tensor_slices.  CPU is reliable and
# only ~30-40% slower on M2 for this dataset size.
tf.config.set_visible_devices([], 'GPU')

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, confusion_matrix)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve   # older sklearn fallback

from models_v2.multimodal_model import build_multimodal_model, save_model

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(ROOT, "data_clean")
METADATA_CSV   = os.path.join(DATA_DIR, "metadata.csv")
SAVED_MODEL_DIR= os.path.join(ROOT, "models_v2", "saved_model")
CHECKPOINT_H5  = os.path.join(ROOT, "models_v2", "best_v2_ckpt.h5")
LOG_PATH       = os.path.join(ROOT, "models_v2", "training_logs_v2.json")
SCALER_PKL     = os.path.join(ROOT, "models_v2", "metadata_scaler.pkl")
IMAGE_SIZE     = (224, 224)
SEED           = 42

CANCER_FOLDER     = "zzz_cancer"
NONCANCER_FOLDER  = "aaa_non_cancer"
LABEL_MAP         = {CANCER_FOLDER: 1, NONCANCER_FOLDER: 0}

os.makedirs(os.path.join(ROOT, "models_v2"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Binary focal loss for handling class imbalance.

    Reference: Lin et al., 2017 — "Focal Loss for Dense Object Detection"

    Args:
        gamma : Focusing parameter (≥0). Higher values down-weight easy
                samples more aggressively. Default 2.0.
        alpha : Weighting for the positive (cancer) class.  Default 0.25.

    Returns:
        Keras-compatible loss function.
    """
    def _focal_loss(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulator = tf.pow(1.0 - p_t, gamma)
        alpha_t   = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * modulator * bce)

    _focal_loss.__name__ = f"focal_loss_g{gamma}_a{alpha}"
    return _focal_loss


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED LOGGING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_extended_metrics(y_true: np.ndarray,
                              y_prob: np.ndarray,
                              threshold: float = 0.5) -> dict:
    """
    Compute sensitivity, specificity, AUC, and calibration for a set of
    binary predictions.

    Args:
        y_true    : Integer ground-truth labels (0 or 1).
        y_prob    : Sigmoid probabilities in [0, 1].
        threshold : Decision threshold.

    Returns:
        Dict with keys: auc, sensitivity, specificity, ppv, npv,
                        calibration_fraction_of_positives (list),
                        calibration_mean_predicted (list).
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv         = tp / (tp + fp + 1e-8)
    npv         = tn / (tn + fn + 1e-8)
    auc         = float(roc_auc_score(y_true, y_prob))

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10,
                                             strategy="uniform")
    return {
        "auc"         : round(auc,         4),
        "sensitivity" : round(sensitivity, 4),
        "specificity" : round(specificity, 4),
        "ppv"         : round(ppv,         4),
        "npv"         : round(npv,         4),
        "calibration_fraction_of_positives": frac_pos.tolist(),
        "calibration_mean_predicted"       : mean_pred.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1.  METADATA CSV  (generate synthetic if absent)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_meta_row(label: int, rng: np.random.Generator) -> dict:
    """Draw one row of synthetic metadata — label-INDEPENDENT distribution.

    We intentionally use the same marginal distribution regardless of label
    so the model CANNOT memorise a metadata→label shortcut and is forced to
    learn from the image branch.  Using biased (label-correlated) metadata
    with a synthetic dataset causes the model to achieve perfect validation
    AUC by ignoring images, which is useless for real screening.
    """
    # Uniform marginals: no correlation with label
    return {
        "age"          : float(np.clip(rng.normal(50, 15), 18, 90)),
        "smoking"      : int(rng.random() < 0.40),
        "alcohol"      : int(rng.random() < 0.35),
        "sun_exposure" : float(rng.uniform(0, 8)),
    }


def ensure_metadata_csv() -> str:
    """
    Return path to the metadata CSV.  If none exists, generate a synthetic
    template and save it to METADATA_CSV so researchers can inspect/replace it.
    """
    if os.path.exists(METADATA_CSV):
        print(f"  ✅ Metadata CSV found: {os.path.relpath(METADATA_CSV, ROOT)}")
        return METADATA_CSV

    print("  ⚠️  No metadata.csv found — generating SYNTHETIC metadata.")
    print(f"     Replace '{os.path.relpath(METADATA_CSV, ROOT)}' with real patient records"
          " before publishing results.")

    rng = np.random.default_rng(SEED)
    rows = []
    for split in ("train", "val"):
        for class_folder, label in LABEL_MAP.items():
            folder_path = os.path.join(DATA_DIR, split, class_folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                row = {
                    "split"       : split,
                    "class_folder": class_folder,
                    "filename"    : fname,
                    "label"       : label,
                    "synthetic"   : True,   # flag — never remove this column
                }
                row.update(_synthetic_meta_row(label, rng))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(METADATA_CSV, index=False)
    print(f"  💾 Synthetic CSV saved → {os.path.relpath(METADATA_CSV, ROOT)}"
          f"  ({len(df)} rows)")
    return METADATA_CSV


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def load_split(df: pd.DataFrame,
               split: str,
               scaler: StandardScaler | None = None,
               fit_scaler: bool = False,
               augment: bool = False
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Load images + metadata for one split.

    Args:
        df         : Full metadata DataFrame.
        split      : "train" | "val"
        scaler     : Pre-fitted StandardScaler (pass None for train split).
        fit_scaler : Fit a new StandardScaler on this split's metadata.
        augment    : Apply random augmentations (train split only).

    Returns:
        images   : float32 (N, 224, 224, 3)
        metadata : float32 (N, 4)  scaled
        labels   : int (N,)
        scaler   : fitted StandardScaler (same object passed or new one)
    """
    subset = df[df["split"] == split].reset_index(drop=True)

    images_list, meta_list, label_list = [], [], []
    skipped = 0

    for _, row in subset.iterrows():
        img_path = os.path.join(
            DATA_DIR, row["split"], row["class_folder"], row["filename"]
        )
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        # NOTE: Do NOT divide by 255 here.  EfficientNetB0 includes its own
        # internal Rescaling + Normalization preprocessing layers, so it expects
        # raw pixel values in [0, 255].  Dividing here causes a double-rescaling
        # that collapses all images to ~0, making the CNN branch non-discriminative.
        img = img.astype("float32")   # keep in [0, 255] range

        if augment:
            img = _augment(img)

        images_list.append(img)
        meta_list.append([
            float(row["age"]),
            float(row["smoking"]),
            float(row["alcohol"]),
            float(row["sun_exposure"]),
        ])
        label_list.append(int(row["label"]))

    if skipped:
        print(f"  ⚠️  Skipped {skipped} unreadable images in '{split}'")

    images   = np.array(images_list,  dtype="float32")
    metadata = np.array(meta_list,    dtype="float32")
    labels   = np.array(label_list,   dtype=int)

    # Scale metadata
    if fit_scaler:
        scaler = StandardScaler()
        metadata = scaler.fit_transform(metadata)
    else:
        if scaler is None:
            raise ValueError("Must pass a fitted scaler for non-train splits")
        metadata = scaler.transform(metadata)

    return images, metadata, labels, scaler


def _augment(img: np.ndarray) -> np.ndarray:
    """
    Apply deterministic-random augmentations on a single (224,224,3) image.
    Expects input in [0, 255] float32 range (raw pixels).
    """
    rng = np.random.default_rng()   # new seed each call → stochastic

    # Horizontal flip
    if rng.random() > 0.5:
        img = img[:, ::-1, :]

    # Vertical flip (less aggressive — oral lesions can appear anywhere)
    if rng.random() > 0.8:
        img = img[::-1, :, :]

    # Random brightness ±10 %
    factor = rng.uniform(0.90, 1.10)
    img    = np.clip(img * factor, 0.0, 255.0)   # clamp to [0, 255]

    # Random 90° rotation
    k = rng.integers(0, 4)
    if k > 0:
        img = np.rot90(img, k)

    return img.astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  KERAS  tf.data  PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(images: np.ndarray,
                 metadata: np.ndarray,
                 labels: np.ndarray,
                 batch_size: int,
                 shuffle: bool = False) -> tf.data.Dataset:
    """Wrap numpy arrays as a batched tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices(
        ({"image_input": images, "metadata_input": metadata},
         labels.astype("float32"))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks(phase: int, learning_rate: float) -> list:
    """
    Return a list of Keras callbacks appropriate for the given training phase.

    Args:
        phase         : 1 (warm-up) or 2 (fine-tune)
        learning_rate : Current base LR (used to set ReduceLR floor).
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_auc",
            patience             = 8 if phase == 1 else 5,
            mode                 = "max",
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = CHECKPOINT_H5,
            monitor        = "val_auc",
            mode           = "max",
            save_best_only = True,
            verbose        = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor   = "val_auc",
            factor    = 0.5,
            patience  = 4,
            mode      = "max",
            min_lr    = learning_rate * 1e-3,
            verbose   = 1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(model: tf.keras.Model,
              train_ds: tf.data.Dataset,
              val_ds: tf.data.Dataset,
              class_weights: dict,
              epochs: int,
              learning_rate: float,
              phase: int,
              history_acc: list,
              use_focal_loss: bool = False) -> None:
    """
    Compile and train for one phase.  Appends epoch logs to history_acc.

    Args:
        model        : The (possibly partially unfrozen) Keras model.
        train_ds     : Training tf.data.Dataset.
        val_ds       : Validation tf.data.Dataset.
        class_weights: {0: w0, 1: w1} dict for imbalance handling.
        epochs       : Max epochs for this phase.
        learning_rate: Adam LR.
        phase        : 1 or 2 (used to label history entries).
        history_acc  : Mutable list; epoch dicts are appended here.
        use_focal_loss: If True, compile with focal loss instead of BCE.
    """
    loss_fn = focal_loss() if use_focal_loss else "binary_crossentropy"
    if use_focal_loss:
        print(f"  Loss function : Focal Loss (gamma=2.0, alpha=0.25)")

    model.compile(
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss      = loss_fn,
        metrics   = [
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    print(f"\n  Trainable parameters: {model.count_params():,}")
    cbs   = build_callbacks(phase, learning_rate)
    hist  = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = epochs,
        class_weight    = class_weights,
        callbacks       = cbs,
        verbose         = 1,
    )

    for i, (loss, acc, auc, prec, rec,
            val_loss, val_acc, val_auc, val_prec, val_rec) in enumerate(zip(
            hist.history["loss"],
            hist.history["accuracy"],
            hist.history["auc"],
            hist.history["precision"],
            hist.history["recall"],
            hist.history["val_loss"],
            hist.history["val_accuracy"],
            hist.history["val_auc"],
            hist.history["val_precision"],
            hist.history["val_recall"],
    )):
        history_acc.append({
            "phase"        : phase,
            "epoch"        : i + 1,
            "loss"         : round(float(loss), 5),
            "accuracy"     : round(float(acc),  4),
            "auc"          : round(float(auc),  4),
            "precision"    : round(float(prec), 4),
            "recall"       : round(float(rec),  4),
            "val_loss"     : round(float(val_loss),  5),
            "val_accuracy" : round(float(val_acc),   4),
            "val_auc"      : round(float(val_auc),   4),
            "val_precision": round(float(val_prec),  4),
            "val_recall"   : round(float(val_rec),   4),
        })


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(epochs_phase1: int = 30,
         epochs_phase2: int = 20,
         batch_size: int    = 16,
         use_focal_loss: bool = False) -> None:

    total_start = time.time()
    print("\n" + "=" * 65)
    print("  CuraLens v2 — Multi-Modal Training Pipeline")
    print(f"  {tf.__version__ = }")
    print("=" * 65)

    # ── 6.1  Metadata ──────────────────────────────────────────────────────
    print("\n[1/6] Metadata …")
    ensure_metadata_csv()
    df = pd.read_csv(METADATA_CSV)
    if "synthetic" in df.columns and df["synthetic"].any():
        print("  ⚠️  Training on SYNTHETIC metadata."
              " Results NOT suitable for clinical claims.")

    # Validate required columns
    required = {"split", "class_folder", "filename",
                "label", "age", "smoking", "alcohol", "sun_exposure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {missing}")

    # ── 6.2  Load images + metadata ────────────────────────────────────────
    print("\n[2/6] Loading images …")
    X_img_tr, X_meta_tr, y_tr, scaler = load_split(
        df, "train", fit_scaler=True, augment=True
    )
    X_img_val, X_meta_val, y_val, _   = load_split(
        df, "val", scaler=scaler, fit_scaler=False, augment=False
    )

    print(f"  Train  : {len(y_tr):>4} samples"
          f"  (cancer={y_tr.sum()}, non-cancer={(1-y_tr).sum()})")
    print(f"  Val    : {len(y_val):>4} samples"
          f"  (cancer={y_val.sum()}, non-cancer={(1-y_val).sum()})")

    # ── 6.3  Class weights ─────────────────────────────────────────────────
    print("\n[3/6] Computing class weights …")
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weights = {0: float(weights[0]), 1: float(weights[1])}
    print(f"  Class weights → {class_weights}")

    # ── 6.4  tf.data pipelines ─────────────────────────────────────────────
    print("\n[4/6] Building tf.data pipelines …")
    train_ds = make_dataset(X_img_tr, X_meta_tr, y_tr,  batch_size, shuffle=True)
    val_ds   = make_dataset(X_img_val, X_meta_val, y_val, batch_size, shuffle=False)

    # ── 6.5  Build model ───────────────────────────────────────────────────
    print("\n[5/6] Building model …")
    model = build_multimodal_model(trainable_cnn=False, learning_rate=1e-4)
    print(f"  Architecture : CuraLens_v2_MultiModal")
    print(f"  Parameters   : {model.count_params():,}")

    history_log: list = []

    # ── 6.6  Phase 1 — warm-up (CNN frozen) ───────────────────────────────
    print("\n" + "─" * 65)
    print("  PHASE 1 — Warm-up  (EfficientNet frozen, LR=1e-4)")
    print("─" * 65)
    phase1_start = time.time()
    run_phase(model, train_ds, val_ds, class_weights,
              epochs_phase1, 1e-4, phase=1, history_acc=history_log,
              use_focal_loss=use_focal_loss)
    print(f"  Phase 1 complete  ({time.time()-phase1_start:.0f}s)")

    # ── 6.7  Phase 2 — fine-tune top 20 EfficientNet layers ───────────────
    print("\n" + "─" * 65)
    print("  PHASE 2 — Fine-tuning  (top-20 EfficientNet layers, LR=1e-5)")
    print("─" * 65)

    # When EfficientNetB0 is built with input_tensor=..., its layers are
    # inlined directly into the outer model (no nested sub-model wrapper).
    # Identify CNN layers by their name prefix; exclude the metadata branch
    # and the fusion/classification head.
    HEAD_PREFIXES = (
        "metadata_input", "image_input",
        "img_proj", "img_dropout",
        "meta_bn", "meta_dense", "meta_dropout",
        "fusion",
        "head_dense", "head_dropout",
        "cancer_prob",
    )

    cnn_layers = [
        layer for layer in model.layers
        if not any(layer.name.startswith(p) for p in HEAD_PREFIXES)
    ]

    # Ensure all CNN layers are frozen first (in case Phase 1 partially
    # changed trainability), then unfreeze the tail 20.
    for layer in cnn_layers:
        layer.trainable = False
    for layer in cnn_layers[-20:]:
        layer.trainable = True

    unfrozen = sum(1 for l in cnn_layers if l.trainable)
    print(f"  EfficientNet layers (total / unfrozen): {len(cnn_layers)} / {unfrozen}")

    phase2_start = time.time()
    run_phase(model, train_ds, val_ds, class_weights,
              epochs_phase2, 1e-5, phase=2, history_acc=history_log,
              use_focal_loss=use_focal_loss)
    print(f"  Phase 2 complete  ({time.time()-phase2_start:.0f}s)")

    # ── 6.8  Save model + logs ─────────────────────────────────────────────
    print("\n[6/6] Saving …")
    save_model(model, SAVED_MODEL_DIR)

    # Save the fitted scaler so that inference uses identical normalisation.
    # web_app.py (and any other inference code) should load this .pkl instead
    # of relying on hard-coded population reference stats.
    with open(SCALER_PKL, "wb") as _f:
        pickle.dump(scaler, _f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  💾 Scaler   → {os.path.relpath(SCALER_PKL, ROOT)}")

    log_payload = {
        "training_date"     : time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture"      : "CuraLens_v2_MultiModal (EfficientNetB0 + Dense metadata)",
        "total_train_samples": int(len(y_tr)),
        "total_val_samples"  : int(len(y_val)),
        "class_weights"     : class_weights,
        "scaler_mean"       : scaler.mean_.tolist(),
        "scaler_scale"      : scaler.scale_.tolist(),
        "metadata_synthetic": bool(
            "synthetic" in df.columns and df["synthetic"].any()
        ),
        "phases": {
            "phase1": {"epochs_requested": epochs_phase1, "lr": 1e-4},
            "phase2": {"epochs_requested": epochs_phase2, "lr": 1e-5,
                       "unfrozen_efficientnet_layers": 20},
        },
        "history": history_log,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log_payload, f, indent=2)
    print(f"  💾 Training log → {os.path.relpath(LOG_PATH, ROOT)}")

    # ── 6.9  Final metrics ─────────────────────────────────────────────────
    # Pick the best val_auc epoch from logs
    best = max(history_log, key=lambda e: e["val_auc"])

    elapsed = time.time() - total_start
    print("\n" + "=" * 65)
    print("  Training Complete")
    print(f"  Total time       : {elapsed/60:.1f} min  ({elapsed:.0f}s)")
    print(f"  Best epoch       : Phase {best['phase']}, Epoch {best['epoch']}")
    print("─" * 65)
    print(f"  Val Accuracy     : {best['val_accuracy']:.4f}")
    print(f"  Val Sensitivity  : {best['val_recall']:.4f}  ← most critical")
    print(f"  Val Precision    : {best['val_precision']:.4f}")
    print(f"  Val AUC          : {best['val_auc']:.4f}")
    print("─" * 65)
    print(f"  Class weights    : {class_weights}")
    if log_payload["metadata_synthetic"]:
        print()
        print("  ⚠️  REMINDER: Trained on SYNTHETIC metadata.")
        print("      Collect real patient records and retrain before")
        print("      making any clinical or research claims.")
    print("=" * 65)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SKIN TRAINING  (new, independent — does not touch existing oral code)
# ─────────────────────────────────────────────────────────────────────────────

SKIN_DATA_DIR     = os.path.join(ROOT, "skin_dataset_resized")
SKIN_SAVED_DIR    = os.path.join(ROOT, "models_v2", "skin_saved_model")
SKIN_CKPT_H5      = os.path.join(ROOT, "models_v2", "best_skin_ckpt.h5")
SKIN_LOG_PATH     = os.path.join(ROOT, "models_v2", "training_logs_skin.json")
SKIN_LABEL_MAP    = {"benign": 0, "malignant": 1}


def _scan_skin_split(split_dir: str) -> tuple[list, np.ndarray]:
    """
    Scan <split_dir>/{benign,malignant}/ and return (file_paths, labels)
    WITHOUT loading any pixel data — RAM cost is a few KB regardless of
    dataset size.
    """
    paths, labels = [], []
    for cls_name, label in sorted(SKIN_LABEL_MAP.items()):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            paths.append(os.path.join(cls_dir, fname))
            labels.append(label)
    return paths, np.array(labels, dtype="int32")


def _decode_skin_image(path: tf.Tensor, augment_flag: tf.Tensor) -> tf.Tensor:
    """Decode a single JPEG/PNG from disk and resize to (224,224,3).
    Keeps values in [0,255] — EfficientNetB0 handles rescaling internally."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
    # Light augmentation applied only for training images
    def _augment(x):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_brightness(x, max_delta=0.15)
        x = tf.image.random_contrast(x, lower=0.85, upper=1.15)
        x = tf.clip_by_value(x, 0.0, 255.0)
        return x
    img = tf.cond(augment_flag, lambda: _augment(img), lambda: img)
    return img


def _load_skin_split(split_dir: str,
                     augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all skin images from  <split_dir>/{benign,malignant}/.
    Returns (images float32 (N,224,224,3), labels int (N,)).
    LEGACY — only used for tiny splits; main train_skin() uses lazy loading.
    """
    imgs, labels = [], []
    for cls_name, label in SKIN_LABEL_MAP.items():
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE).astype("float32") / 255.0
            if augment:
                img = _augment(img)
            imgs.append(img)
            labels.append(label)
    return np.array(imgs, dtype="float32"), np.array(labels, dtype=int)


def _synthetic_skin_meta(n: int, labels: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic 6D skin metadata rows matching the SKIN_SCHEMA.

    NOTE: Distributions are intentionally LABEL-INDEPENDENT so the model is
    forced to learn visual features rather than memorising metadata risk scores.
    """
    from utils_v2.metadata_schema import SKIN_FIELDS, _normalize_value
    rows = []
    for _ in labels:   # label ignored — same population distribution for both classes
        raw = {
            "age"                    : float(np.clip(rng.normal(48, 15), 18, 90)),
            "skin_type"              : int(rng.integers(1, 7)),
            "sunburn_history"        : int(rng.integers(0, 20)),
            "outdoor_hours_per_week" : float(rng.uniform(1, 50)),
            "tanning_bed_use"        : int(rng.random() < 0.22),
            "family_history"         : int(rng.random() < 0.20),
        }
        row = [_normalize_value(raw[s.name], s) for s in SKIN_FIELDS]
        rows.append(row)
    return np.array(rows, dtype="float32")


def train_skin(
    epochs_phase1: int   = 30,
    epochs_phase2: int   = 20,
    batch_size: int      = 16,
    use_focal_loss: bool = False,
) -> None:
    """Train the CuraLens Skin multimodal model."""
    from models_v2.skin_model import build_skin_model, save_skin_model

    total_start = time.time()
    print("\n" + "=" * 65)
    print("  CuraLens Skin — Multi-Modal Training Pipeline")
    print("=" * 65)

    os.makedirs(SKIN_SAVED_DIR, exist_ok=True)

    # ── Load images ────────────────────────────────────────────────────
    train_dir = os.path.join(SKIN_DATA_DIR, "train_set")
    val_dir   = os.path.join(SKIN_DATA_DIR, "val_set")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Skin training directory not found: {train_dir}\n"
            "Expected layout: skin_dataset_resized/{{train_set,val_set}}/{{benign,malignant}}/"
        )

    print("\n[1/5] Scanning skin image paths (lazy loading — no RAM spike) …")
    # Only collect file paths + labels; pixel data is read batch-by-batch during
    # training.  Peak RAM = model weights (~200 MB) + 1 batch (~9 MB) instead
    # of the full 12 GB that loading all images at once would require.
    paths_tr,  y_tr  = _scan_skin_split(train_dir)
    paths_val, y_val = _scan_skin_split(val_dir)

    print(f"  Train: {len(y_tr):>4}  (malignant={int(y_tr.sum())}, benign={int((y_tr==0).sum())})")
    print(f"  Val  : {len(y_val):>4}  (malignant={int(y_val.sum())}, benign={int((y_val==0).sum())})")

    # ── Synthetic metadata ─────────────────────────────────────────────
    # Metadata is 6 floats × N rows ≈ 220 KB — perfectly fine in memory.
    print("\n[2/5] Generating synthetic skin metadata …")
    rng = np.random.default_rng(SEED)
    X_meta_tr  = _synthetic_skin_meta(len(y_tr),  y_tr,  rng)
    X_meta_val = _synthetic_skin_meta(len(y_val), y_val, rng)
    print("  ⚠️  Synthetic metadata in use — replace with real records before publication.")

    # ── Class weights ──────────────────────────────────────────────────
    print("\n[3/5] Computing class weights …")
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weights = {0: float(weights[0]), 1: float(weights[1])}
    print(f"  Class weights → {class_weights}")

    # ── tf.data — lazy image decoding ──────────────────────────────────
    print("\n[4/5] Building tf.data pipelines (streaming from disk) …")

    def _make_ds(paths, meta, labels, augment: bool, shuffle: bool) -> tf.data.Dataset:
        aug_tensor = tf.constant(augment)
        img_ds  = tf.data.Dataset.from_tensor_slices(paths).map(
            lambda p: _decode_skin_image(p, aug_tensor),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        meta_ds  = tf.data.Dataset.from_tensor_slices(meta)
        label_ds = tf.data.Dataset.from_tensor_slices(labels.astype("float32"))
        ds = tf.data.Dataset.zip(
            ({"skin_image_input": img_ds, "skin_metadata_input": meta_ds}, label_ds)
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=min(2000, len(labels)), seed=SEED,
                            reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _make_ds(paths_tr,  X_meta_tr,  y_tr,  augment=True,  shuffle=True)
    val_ds   = _make_ds(paths_val, X_meta_val, y_val, augment=False, shuffle=False)

    # ── Build model ────────────────────────────────────────────────────
    print("\n[5/5] Training …")
    model = build_skin_model(trainable_cnn=False, learning_rate=1e-4)

    history_log: list = []

    # Temporarily point checkpoint to skin ckpt
    _orig_ckpt = CHECKPOINT_H5
    skin_cbs_phase1 = [
        tf.keras.callbacks.EarlyStopping("val_auc", patience=8, mode="max",
                                          restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(SKIN_CKPT_H5, "val_auc", mode="max",
                                            save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau("val_auc", factor=0.5, patience=4,
                                              mode="max", min_lr=1e-7, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    loss_fn  = focal_loss() if use_focal_loss else "binary_crossentropy"
    model.compile(
        optimizer = tf.keras.optimizers.legacy.Adam(1e-4),
        loss      = loss_fn,
        metrics   = ["accuracy",
                     tf.keras.metrics.AUC(name="auc"),
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")],
    )
    p1_hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs_phase1,
                        class_weight=class_weights, callbacks=skin_cbs_phase1, verbose=1)

    # Phase 2
    skin_cnn_layers = [l for l in model.layers
                       if not any(l.name.startswith(p) for p in
                                  ["skin_image_input", "skin_metadata_input",
                                   "skin_img_proj", "skin_img_dropout",
                                   "skin_meta", "skin_fusion",
                                   "skin_head", "skin_cancer_prob"])]
    for l in skin_cnn_layers: l.trainable = False
    for l in skin_cnn_layers[-20:]: l.trainable = True

    model.compile(
        optimizer = tf.keras.optimizers.legacy.Adam(1e-5),
        loss      = loss_fn,
        metrics   = ["accuracy",
                     tf.keras.metrics.AUC(name="auc"),
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")],
    )
    skin_cbs_phase2 = [
        tf.keras.callbacks.EarlyStopping("val_auc", patience=5, mode="max",
                                          restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(SKIN_CKPT_H5, "val_auc", mode="max",
                                            save_best_only=True, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_phase2,
              class_weight=class_weights, callbacks=skin_cbs_phase2, verbose=1)

    # Save
    save_skin_model(model, SKIN_SAVED_DIR)

    elapsed = time.time() - total_start
    print(f"\n  ✅ Skin model trained in {elapsed/60:.1f} min → {SKIN_SAVED_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION  (oral only, wraps existing code)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_oral(
    df: pd.DataFrame,
    n_folds: int     = 5,
    batch_size: int  = 16,
    epochs: int      = 20,
    use_focal_loss: bool = False,
) -> dict:
    """
    Stratified K-fold cross-validation for the oral multimodal model.

    Args:
        df            : Full metadata DataFrame (all splits combined).
        n_folds       : Number of folds.
        batch_size    : Batch size.
        epochs        : Max epochs per fold.
        use_focal_loss: Use focal loss.

    Returns:
        Dict with per-fold and aggregated AUC / sensitivity / specificity.
    """
    print(f"\n[CV] Stratified {n_folds}-fold cross-validation starting …")

    # Collect all images + labels
    rows = df.copy()
    labels_all = []
    imgs_all, meta_raw_all = [], []

    rng = np.random.default_rng(SEED)
    for _, row in rows.iterrows():
        split  = row.get("split", "train")
        folder = row.get("class_folder", "aaa_non_cancer")
        fname  = row["filename"]
        fpath  = os.path.join(DATA_DIR, split, folder, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE).astype("float32") / 255.0
        imgs_all.append(img)
        meta_raw_all.append([
            float(row["age"]), float(row["smoking"]),
            float(row["alcohol"]), float(row["sun_exposure"]),
        ])
        labels_all.append(int(row["label"]))

    X_img = np.array(imgs_all, dtype="float32")
    X_meta_raw = np.array(meta_raw_all, dtype="float32")
    y = np.array(labels_all, dtype=int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = []
    loss_fn = focal_loss() if use_focal_loss else "binary_crossentropy"

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_img, y), start=1):
        print(f"\n  ── Fold {fold_idx}/{n_folds} ──")
        scaler = StandardScaler()
        X_meta_tr  = scaler.fit_transform(X_meta_raw[tr_idx])
        X_meta_val = scaler.transform(X_meta_raw[val_idx])

        train_ds = make_dataset(X_img[tr_idx], X_meta_tr,  y[tr_idx],  batch_size, shuffle=True)
        val_ds   = make_dataset(X_img[val_idx], X_meta_val, y[val_idx], batch_size, shuffle=False)

        model = build_multimodal_model(trainable_cnn=False, learning_rate=1e-4)
        model.compile(
            optimizer = tf.keras.optimizers.legacy.Adam(1e-4),
            loss      = loss_fn,
            metrics   = ["accuracy", tf.keras.metrics.AUC(name="auc"),
                         tf.keras.metrics.Recall(name="recall")],
        )
        cbs = [tf.keras.callbacks.EarlyStopping("val_auc", patience=5, mode="max",
                                                  restore_best_weights=True),
               tf.keras.callbacks.TerminateOnNaN()]
        model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  callbacks=cbs, verbose=0)

        y_prob = model.predict(
            {"image_input": X_img[val_idx], "metadata_input": X_meta_val},
            verbose=0
        ).flatten()
        metrics = compute_extended_metrics(y[val_idx], y_prob)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)
        print(f"    AUC={metrics['auc']:.4f}  "
              f"Sens={metrics['sensitivity']:.4f}  "
              f"Spec={metrics['specificity']:.4f}")

    agg = {
        "mean_auc"        : round(float(np.mean([r["auc"] for r in fold_results])), 4),
        "std_auc"         : round(float(np.std( [r["auc"] for r in fold_results])), 4),
        "mean_sensitivity": round(float(np.mean([r["sensitivity"] for r in fold_results])), 4),
        "mean_specificity": round(float(np.mean([r["specificity"] for r in fold_results])), 4),
    }
    print(f"\n[CV] Results: AUC={agg['mean_auc']:.4f}±{agg['std_auc']:.4f}  "
          f"Sens={agg['mean_sensitivity']:.4f}  Spec={agg['mean_specificity']:.4f}")

    cv_output = {"fold_results": fold_results, "aggregate": agg}
    cv_log = os.path.join(ROOT, "models_v2", "cv_results.json")
    with open(cv_log, "w") as f:
        json.dump(cv_output, f, indent=2)
    print(f"[CV] Results saved → {os.path.relpath(cv_log, ROOT)}")
    return cv_output


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CuraLens v2/v3 Multi-Modal Training Pipeline"
    )
    parser.add_argument(
        "--cancer-type", type=str, default="oral_legacy",
        choices=["oral_legacy", "oral", "skin"],
        help=(
            "Which cancer type to train. "
            "'oral_legacy' = existing 4D metadata oral model (default, backward-compat). "
            "'oral' = reserved for future 6D oral retraining. "
            "'skin' = train the skin multimodal model."
        ),
    )
    parser.add_argument(
        "--epochs-phase1", type=int, default=30,
        help="Max epochs for Phase 1 warm-up (default: 30)",
    )
    parser.add_argument(
        "--epochs-phase2", type=int, default=20,
        help="Max epochs for Phase 2 fine-tuning (default: 20)",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size for both phases (default: 16)",
    )
    parser.add_argument(
        "--use-focal-loss", action="store_true",
        help="Use binary focal loss (gamma=2, alpha=0.25) instead of BCE",
    )
    parser.add_argument(
        "--cross-validate", action="store_true",
        help="Run stratified K-fold cross-validation (oral only)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    args = parser.parse_args()

    cancer_type = args.cancer_type

    if cancer_type == "skin":
        train_skin(
            epochs_phase1  = args.epochs_phase1,
            epochs_phase2  = args.epochs_phase2,
            batch_size     = args.batch,
            use_focal_loss = args.use_focal_loss,
        )
    elif args.cross_validate:
        # Cross-validation mode (oral_legacy only for now)
        print("\n[CV] Loading metadata for cross-validation …")
        ensure_metadata_csv()
        df_cv = pd.read_csv(METADATA_CSV)
        cross_validate_oral(
            df            = df_cv,
            n_folds       = args.cv_folds,
            batch_size    = args.batch,
            epochs        = args.epochs_phase1,
            use_focal_loss= args.use_focal_loss,
        )
    else:
        # Default: oral_legacy (unchanged behavior)
        main(
            epochs_phase1  = args.epochs_phase1,
            epochs_phase2  = args.epochs_phase2,
            batch_size     = args.batch,
            use_focal_loss = args.use_focal_loss,
        )
