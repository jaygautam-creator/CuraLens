"""
CuraLens – Skin Model Test-Set Evaluation
==========================================
Evaluates the v1 MobileNetV2 skin model on the held-out test_set/ directory
and saves full clinical metrics to evaluation_outputs/skin_test_metrics.json.

Usage:
    python evaluate_skin.py

Outputs:
    evaluation_outputs/skin_test_metrics.json  ← metrics + confusion matrix
    evaluation_outputs/skin_roc_curve.png      ← ROC curve plot
    evaluation_outputs/skin_confusion_matrix.png
"""

from __future__ import annotations

import os
import json
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import numpy as np

TEST_DIR    = os.path.join(ROOT, "skin_dataset_resized", "test_set")
MODEL_PATH  = os.path.join(ROOT, "models", "skin_model", "skin_screening_model.h5")
OUTPUT_DIR  = os.path.join(ROOT, "evaluation_outputs")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "skin_test_metrics.json")
ROC_PNG     = os.path.join(OUTPUT_DIR, "skin_roc_curve.png")
CM_PNG      = os.path.join(OUTPUT_DIR, "skin_confusion_matrix.png")

LABEL_MAP = {"benign": 0, "malignant": 1}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images(split_dir: str):
    """
    Load all JPEG/PNG images from <split_dir>/{benign,malignant}/.

    Returns:
        images : float32 ndarray  (N, 224, 224, 3) normalised to [0, 1]
        labels : int ndarray      (N,)  0=benign, 1=malignant
        paths  : list[str]        corresponding file paths (for debug)
    """
    from PIL import Image

    images, labels, paths = [], [], []
    for cls_name, label in sorted(LABEL_MAP.items()):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  ⚠️  Class directory not found, skipping: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(cls_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize((224, 224))
                images.append(np.array(img, dtype="float32") / 255.0)
                labels.append(label)
                paths.append(fpath)
            except Exception as e:
                print(f"  ⚠️  Could not load {fpath}: {e}")

    return np.array(images, dtype="float32"), np.array(labels, dtype=int), paths


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def find_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the decision threshold that maximises Youden's J statistic."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    return float(thresholds[best_idx])


def compute_metrics(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: float) -> dict:
    """Compute full clinical metric suite."""
    from sklearn.metrics import roc_auc_score, confusion_matrix

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision   = tp / (tp + fp + 1e-8)
    npv         = tn / (tn + fn + 1e-8)
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    auc         = roc_auc_score(y_true, y_prob)

    return {
        "roc_auc":          round(float(auc),         4),
        "sensitivity":      round(float(sensitivity),  4),
        "specificity":      round(float(specificity),  4),
        "precision":        round(float(precision),    4),
        "npv":              round(float(npv),           4),
        "f1_score":         round(float(f1),            4),
        "accuracy":         round(float(accuracy),      4),
        "optimal_threshold": round(float(threshold),   4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "n_samples":   int(len(y_true)),
        "n_malignant": int(y_true.sum()),
        "n_benign":    int((y_true == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, auc: float) -> None:
    """Save a ROC curve PNG to evaluation_outputs/."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#7c3aed", lw=2,
            label=f"Skin v1 MobileNetV2 (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Skin Model — ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(ROC_PNG, dpi=150)
    plt.close(fig)
    print(f"  💾 ROC curve → {os.path.relpath(ROC_PNG, ROOT)}")


def plot_confusion_matrix(cm: dict) -> None:
    """Save a confusion matrix heatmap PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    data = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    labels = [["TN", "FP"], ["FN", "TP"]]
    cmap = mcolors.LinearSegmentedColormap.from_list("cm_cmap", ["#f0fdf4", "#7c3aed"])

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data, cmap=cmap)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Benign", "Predicted Malignant"])
    ax.set_yticklabels(["Actual Benign", "Actual Malignant"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{data[i, j]}",
                    ha="center", va="center",
                    color="white" if data[i, j] > data.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_title("Skin Model — Confusion Matrix (Test Set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CM_PNG, dpi=150)
    plt.close(fig)
    print(f"  💾 Confusion matrix → {os.path.relpath(CM_PNG, ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import tensorflow as tf

    print("=" * 65)
    print("  CuraLens – Skin Model Test-Set Evaluation")
    print("=" * 65)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading model from {os.path.relpath(MODEL_PATH, ROOT)} …")
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Model not found: {MODEL_PATH}")
        print("     Run  python train_skin.py  or  python train_v2.py --cancer-type skin  first.")
        sys.exit(1)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("  ✅ Model loaded")

    # ── Load test images ────────────────────────────────────────────────────
    print(f"\nLoading test images from {os.path.relpath(TEST_DIR, ROOT)} …")
    if not os.path.isdir(TEST_DIR):
        print(f"  ❌ Test directory not found: {TEST_DIR}")
        sys.exit(1)

    X, y, paths = load_images(TEST_DIR)
    print(f"  Loaded: {len(y)} images  "
          f"(malignant={int(y.sum())}, benign={int((y == 0).sum())})")

    if len(y) < 2:
        print("  ❌ Need at least 2 labelled images to evaluate.")
        sys.exit(1)

    # ── Predict ─────────────────────────────────────────────────────────────
    print("\nRunning predictions …")
    y_prob = model.predict(X, verbose=1).flatten()

    # ── Threshold (Youden's J) ───────────────────────────────────────────────
    threshold = find_youden_threshold(y, y_prob)
    print(f"\n  Optimal threshold (Youden's J): {threshold:.4f}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(y, y_prob, threshold)

    output = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "v1_skin_MobileNetV2",
        "dataset": "skin_dataset_resized/test_set",
        "metrics": metrics,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    try:
        plot_roc(y, y_prob, metrics["roc_auc"])
    except Exception as e:
        print(f"  ⚠️  ROC plot failed: {e}")

    try:
        plot_confusion_matrix(metrics["confusion_matrix"])
    except Exception as e:
        print(f"  ⚠️  Confusion matrix plot failed: {e}")

    # ── Report ────────────────────────────────────────────────────────────────
    cm = metrics["confusion_matrix"]
    print("\n" + "=" * 65)
    print("  Skin Model  –  Test-Set Results")
    print("─" * 65)
    print(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
    print(f"  Sensitivity  : {metrics['sensitivity']:.4f}  ← fraction of malignant correctly identified")
    print(f"  Specificity  : {metrics['specificity']:.4f}  ← fraction of benign correctly identified")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  NPV          : {metrics['npv']:.4f}")
    print(f"  F1 Score     : {metrics['f1_score']:.4f}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Threshold    : {metrics['optimal_threshold']:.4f}  (Youden's J)")
    print("─" * 65)
    print(f"  Confusion Matrix:  TN={cm['tn']}   FP={cm['fp']}")
    print(f"                     FN={cm['fn']}   TP={cm['tp']}")
    print("=" * 65)
    print(f"\n  💾 Metrics saved → {os.path.relpath(OUTPUT_JSON, ROOT)}")


if __name__ == "__main__":
    main()
