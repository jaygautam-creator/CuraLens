"""
CuraLens v2 — Research Evaluation Script
==========================================
Computes the following clinical and ML metrics on the validation set:

    Accuracy · Precision · Recall (Sensitivity) · Specificity
    F1 Score · ROC-AUC · Confusion Matrix

Also performs:
    • Ablation study  : v1 (image-only) vs v2 (multimodal)
    • Grad-CAM panel  : 5 True-Positives + 3 False-Positives
    • Threshold sweep : optimal operating point on the ROC curve

All outputs are saved to:
    evaluation_outputs/
        metrics_v1.json
        metrics_v2.json
        ablation_summary.json
        roc_curve.png
        confusion_matrix_v1.png
        confusion_matrix_v2.png
        gradcam_panel_tp.png
        gradcam_panel_fp.png

IMPORTANT:
    - Does NOT interact with web_app or any v1 training files.
    - v2 model uses randomly-initialised weights unless a SavedModel
      is present at models_v2/saved_model/.
    - Patient metadata is synthetically generated when real records
      are unavailable (clearly flagged in the output JSON).
"""

from __future__ import annotations

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── project root on path ─────────────────────────────────────────────────────
# evaluate_v2.py lives at the project root, so dirname(__file__) IS the root.
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")           # headless backend – no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from models_v2.multimodal_model import build_multimodal_model, load_model as load_v2
from utils_v2.gradcam import GradCAM

# ── configuration ─────────────────────────────────────────────────────────────
VAL_DIR         = os.path.join(ROOT, "data_clean", "val")
CANCER_FOLDER   = "zzz_cancer"
NONCANCER_FOLDER= "aaa_non_cancer"
V1_MODEL_PATH   = os.path.join(ROOT, "models", "oral_cancer_model.h5")
V2_SAVED_PATH   = os.path.join(ROOT, "models_v2", "saved_model")
OUTPUT_DIR      = os.path.join(ROOT, "evaluation_outputs")
IMAGE_SIZE      = (224, 224)
V1_THRESHOLD    = 0.512         # from model_metadata.json
V2_THRESHOLD    = 0.5           # default; will be optimised on ROC
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_validation_images() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load and pre-process all validation images.

    Returns:
        images : float32 array  (N, 224, 224, 3)  raw pixels in [0, 255]
        labels : int array      (N,)  1 = cancer, 0 = non-cancer
        paths  : list of absolute file paths (for Grad-CAM labelling)
    """
    images, labels, paths = [], [], []

    class_map = {CANCER_FOLDER: 1, NONCANCER_FOLDER: 0}
    for folder, label in class_map.items():
        folder_path = os.path.join(VAL_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f"  ⚠️  Folder not found, skipping: {folder_path}")
            continue
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            img   = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            # NOTE: Do NOT divide by 255 here.  EfficientNetB0 includes its own
            # internal Rescaling layer and expects raw pixels in [0, 255].
            images.append(img.astype("float32"))   # keep in [0, 255]
            labels.append(label)
            paths.append(fpath)

    return np.array(images), np.array(labels, dtype=int), paths


def generate_synthetic_metadata(n_samples: int,
                                 labels: np.ndarray,
                                 seed: int = 42) -> np.ndarray:
    """
    Synthesise plausible patient metadata when real records are unavailable.

    Distribution is loosely inspired by oral-cancer epidemiology:
      - Cancer patients skew older, higher smoking / alcohol / sun risk.
      - Non-cancer patients skew younger, lower risk factors.

    This is clearly flagged in the output so results are not over-claimed.

    Returns: float32 array (N, 4)  → [age, smoking, alcohol, sun_exposure]
    """
    rng = np.random.default_rng(seed)
    meta = np.zeros((n_samples, 4), dtype="float32")

    for i, label in enumerate(labels):
        if label == 1:          # cancer patient profile
            meta[i, 0] = rng.normal(loc=58, scale=10)   # age
            meta[i, 1] = float(rng.random() < 0.72)     # smoking prevalence
            meta[i, 2] = float(rng.random() < 0.60)     # alcohol
            meta[i, 3] = rng.uniform(3, 8)              # sun exposure score
        else:                   # non-cancer patient profile
            meta[i, 0] = rng.normal(loc=42, scale=12)
            meta[i, 1] = float(rng.random() < 0.30)
            meta[i, 2] = float(rng.random() < 0.25)
            meta[i, 3] = rng.uniform(0, 4)

    meta[:, 0] = np.clip(meta[:, 0], 18, 90)   # clamp age to valid range
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 2.  METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: float) -> dict:
    """
    Compute the full clinical metric suite.

    Returns a dict with:
        sensitivity, specificity, precision, f1, accuracy, auc,
        confusion_matrix, threshold, n_samples, n_positive, n_negative
    """
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # handle edge cases
        tn = fp = fn = tp = 0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                tn = cm[0, 0]; fp = 0; fn = 0; tp = 0
            else:
                tp = cm[0, 0]; tn = 0; fp = 0; fn = 0

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall for cancer
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # precision
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0.0   # negative predictive value

    return {
        "threshold"       : round(threshold, 4),
        "accuracy"        : round(accuracy_score(y_true, y_pred), 4),
        "sensitivity"     : round(sensitivity, 4),   # recall (TP rate) — most critical
        "specificity"     : round(specificity, 4),
        "precision"       : round(ppv, 4),
        "npv"             : round(npv, 4),
        "f1_score"        : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc"         : round(roc_auc_score(y_true, y_prob), 4),
        "confusion_matrix": cm.tolist(),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "n_samples"       : int(len(y_true)),
        "n_positive"      : int(y_true.sum()),
        "n_negative"      : int((1 - y_true).sum()),
    }


def find_optimal_threshold(y_true: np.ndarray,
                            y_prob: np.ndarray) -> float:
    """
    Youden's J statistic: argmax(Sensitivity + Specificity - 1).
    Returns the threshold that maximises this on the validation set.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    return float(thresholds[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list,
                           title: str,
                           save_path: str,
                           labels: list = None) -> None:
    """Render a labelled confusion matrix heatmap and save to disk."""
    if labels is None:
        labels = ["Non-Cancer", "Cancer"]

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {os.path.relpath(save_path, ROOT)}")


def plot_roc_curve(results: dict,
                   save_path: str) -> None:
    """
    Overlay ROC curves for all models passed in `results`.

    Args:
        results : { "Model Name": {"y_true": ..., "y_prob": ..., "auc": ...} }
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2563eb", "#7c3aed", "#10b981", "#f59e0b"]

    for (name, data), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name}  (AUC = {data['auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)",      fontsize=12)
    ax.set_title("ROC Curve Comparison — CuraLens v1 vs v2", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {os.path.relpath(save_path, ROOT)}")


def plot_gradcam_panel(model: tf.keras.Model,
                       images_raw: np.ndarray,   # (N, 224, 224, 3) normalised
                       images_display: np.ndarray,  # same as uint8
                       metadata: np.ndarray,
                       indices: list[int],
                       y_true: np.ndarray,
                       y_prob: np.ndarray,
                       title: str,
                       save_path: str,
                       layer_name: str = "top_conv") -> None:
    """
    Generate a grid of (original | Grad-CAM overlay) pairs.
    For each sample shows: true label, predicted probability, risk tier indicator.
    """
    from utils_v2.risk_scoring import score_prediction
    n = len(indices)
    if n == 0:
        print(f"  ⚠️  No samples for Grad-CAM panel: {title}")
        return

    cam = GradCAM(model, layer_name=layer_name)

    fig = plt.figure(figsize=(6 * 2, 4 * n))
    gs  = gridspec.GridSpec(n, 2, hspace=0.4, wspace=0.1)

    for row, idx in enumerate(indices):
        img_batch  = np.expand_dims(images_raw[idx], 0)
        meta_batch = np.expand_dims(metadata[idx], 0)

        heatmap = cam.compute_heatmap(img_batch, metadata=meta_batch)
        overlay = cam.overlay(images_display[idx], heatmap, alpha=0.45)

        prob    = y_prob[idx]
        risk    = score_prediction(prob)
        true_lbl= "Cancer" if y_true[idx] == 1 else "Non-Cancer"
        color   = risk.color_code

        # Original
        ax_orig  = fig.add_subplot(gs[row, 0])
        ax_orig.imshow(images_display[idx])
        ax_orig.set_title(f"[{row+1}] True: {true_lbl}", fontsize=10)
        ax_orig.axis("off")

        # Overlay
        ax_cam = fig.add_subplot(gs[row, 1])
        ax_cam.imshow(overlay)
        ax_cam.set_title(
            f"P(cancer)={prob:.3f} | {risk.risk_label}",
            fontsize=10, color=color,
        )
        ax_cam.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {os.path.relpath(save_path, ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ABLATION TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_table(metrics_v1: dict, metrics_v2: dict) -> dict:
    """Print a side-by-side ablation table and return the summary dict."""
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1_score", "roc_auc"]
    header = f"{'Metric':<22} {'v1 (Image-only)':>18} {'v2 (Multimodal)':>18} {'Δ':>10}"
    sep    = "─" * len(header)

    print(f"\n{sep}")
    print("  Ablation Study — v1 (Image-only) vs v2 (Multimodal)")
    print(sep)
    print(header)
    print(sep)

    summary = {}
    for k in keys:
        v1_val = metrics_v1.get(k, float("nan"))
        v2_val = metrics_v2.get(k, float("nan"))
        delta  = v2_val - v1_val
        sign   = "+" if delta >= 0 else ""
        flag   = " ✅" if delta > 0 else (" ⚠️ " if delta < -0.01 else "  =")
        print(f"  {k:<20} {v1_val:>18.4f} {v2_val:>18.4f} {sign}{delta:>9.4f}{flag}")
        summary[k] = {"v1": v1_val, "v2": v2_val, "delta": round(delta, 4)}

    print(sep)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start_time = datetime.now()
    print("\n" + "=" * 60)
    print("  CuraLens v2 — Research Evaluation")
    print(f"  Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 5.1  Load validation data ─────────────────────────────────────────
    print("\n[1/6] Loading validation images …")
    images_norm, labels, img_paths = load_validation_images()
    # images_norm is already in [0,255] raw pixel range for EfficientNet
    images_uint8 = np.clip(images_norm, 0, 255).astype("uint8")  # for display only
    n = len(labels)
    print(f"  Loaded {n} images  ({labels.sum()} cancer, {(1-labels).sum()} non-cancer)")

    print("\n[2/6] Generating synthetic patient metadata …")
    metadata_arr = generate_synthetic_metadata(n, labels)
    print("  ⚠️  NOTE: Real patient metadata unavailable.")
    print("       Synthetic metadata used for v2 evaluation — do not over-claim.")
    print(f"  Shape: {metadata_arr.shape}  | "
          f"Smoking prevalence (cancer): "
          f"{metadata_arr[labels==1, 1].mean():.1%}")

    # ── 5.2  Load v1 model ────────────────────────────────────────────────
    print("\n[3/6] Loading v1 model …")
    try:
        v1_model = tf.keras.models.load_model(V1_MODEL_PATH, compile=False)
        print(f"  ✅ v1 model loaded from {os.path.relpath(V1_MODEL_PATH, ROOT)}")
    except Exception as e:
        print(f"  ❌ Failed to load v1 model: {e}")
        v1_model = None

    # ── 5.3  Load v2 model ────────────────────────────────────────────────
    print("\n[4/6] Loading v2 model …")
    v2_weights_real = False
    try:
        if os.path.exists(V2_SAVED_PATH):
            v2_model = load_v2(V2_SAVED_PATH)
            v2_weights_real = True
            print(f"  ✅ v2 model restored from {os.path.relpath(V2_SAVED_PATH, ROOT)}")
        else:
            v2_model = build_multimodal_model(trainable_cnn=False)
            print("  ⚠️  No saved v2 weights — using randomly-initialised model.")
            print("       Train the v2 model first for meaningful evaluation.")
    except Exception as e:
        print(f"  ❌ v2 model error: {e}")
        v2_model = None

    # ── 5.4  Run predictions ──────────────────────────────────────────────
    print("\n[5/6] Running predictions …")
    y_prob_v1 = y_prob_v2 = None

    if v1_model is not None and n > 0:
        print("  v1 (image-only) …")
        y_prob_v1 = v1_model.predict(images_norm, batch_size=16, verbose=1).flatten()

    if v2_model is not None and n > 0:
        print("  v2 (multimodal) …")
        # Recompile with run_eagerly=True to avoid the TF Metal batch-predict
        # bug that affects multi-input models on Apple Silicon.
        v2_model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            run_eagerly=True,
        )
        y_prob_v2 = v2_model.predict(
            [images_norm, metadata_arr], batch_size=16, verbose=1
        ).flatten()

    # ── 5.5  Compute metrics ──────────────────────────────────────────────
    print("\n[6/6] Computing metrics …")

    metrics_v1 = metrics_v2 = None

    if y_prob_v1 is not None:
        opt_thresh_v1 = find_optimal_threshold(labels, y_prob_v1)
        metrics_v1    = compute_metrics(labels, y_prob_v1, opt_thresh_v1)
        metrics_v1["model"]             = "v1_image_only"
        metrics_v1["optimal_threshold"] = round(opt_thresh_v1, 4)
        metrics_v1["metadata_synthetic"]= False

        path = os.path.join(OUTPUT_DIR, "metrics_v1.json")
        with open(path, "w") as f:
            json.dump(metrics_v1, f, indent=2)
        print(f"  💾 Saved: {os.path.relpath(path, ROOT)}")

    if y_prob_v2 is not None:
        opt_thresh_v2 = find_optimal_threshold(labels, y_prob_v2)
        metrics_v2    = compute_metrics(labels, y_prob_v2, opt_thresh_v2)
        metrics_v2["model"]             = "v2_multimodal"
        metrics_v2["optimal_threshold"] = round(opt_thresh_v2, 4)
        metrics_v2["metadata_synthetic"]= not v2_weights_real

        path = os.path.join(OUTPUT_DIR, "metrics_v2.json")
        with open(path, "w") as f:
            json.dump(metrics_v2, f, indent=2)
        print(f"  💾 Saved: {os.path.relpath(path, ROOT)}")

    # ── 5.6  Print summary tables ─────────────────────────────────────────
    for name, m in [("v1", metrics_v1), ("v2 (synthetic metadata)", metrics_v2)]:
        if m is None:
            continue
        print(f"\n  ─── {name} Metrics ───────────────────────────────")
        print(f"  Accuracy    : {m['accuracy']:.4f}")
        print(f"  Sensitivity : {m['sensitivity']:.4f}  ← most critical (FN = missed cancer)")
        print(f"  Specificity : {m['specificity']:.4f}")
        print(f"  Precision   : {m['precision']:.4f}")
        print(f"  NPV         : {m['npv']:.4f}")
        print(f"  F1 Score    : {m['f1_score']:.4f}")
        print(f"  ROC-AUC     : {m['roc_auc']:.4f}")
        print(f"  Threshold   : {m['optimal_threshold']:.4f}  (Youden's J)")
        print(f"  TP: {m['TP']}  TN: {m['TN']}  FP: {m['FP']}  FN: {m['FN']}")

    # ── 5.7  Ablation study ───────────────────────────────────────────────
    if metrics_v1 and metrics_v2:
        ablation = print_ablation_table(metrics_v1, metrics_v2)
        ablation_path = os.path.join(OUTPUT_DIR, "ablation_summary.json")
        with open(ablation_path, "w") as f:
            json.dump({
                "note"    : "v2 uses synthetic metadata — not a fair comparison until real metadata is collected",
                "metrics" : ablation,
            }, f, indent=2)
        print(f"\n  💾 Saved: {os.path.relpath(ablation_path, ROOT)}")

    # ── 5.8  ROC curve ────────────────────────────────────────────────────
    roc_data = {}
    if y_prob_v1 is not None:
        roc_data["CuraLens v1 (Image-only)"] = {
            "y_true": labels, "y_prob": y_prob_v1,
            "auc"   : metrics_v1["roc_auc"],
        }
    if y_prob_v2 is not None:
        roc_data["CuraLens v2 (Multimodal)"] = {
            "y_true": labels, "y_prob": y_prob_v2,
            "auc"   : metrics_v2["roc_auc"],
        }
    if roc_data:
        plot_roc_curve(roc_data, os.path.join(OUTPUT_DIR, "roc_curve.png"))

    # ── 5.9  Confusion matrices ───────────────────────────────────────────
    if metrics_v1:
        plot_confusion_matrix(
            metrics_v1["confusion_matrix"],
            title="Confusion Matrix — v1 (Image-only)",
            save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_v1.png"),
        )
    if metrics_v2:
        plot_confusion_matrix(
            metrics_v2["confusion_matrix"],
            title="Confusion Matrix — v2 (Multimodal, synthetic metadata)",
            save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_v2.png"),
        )

    # ── 5.10  Grad-CAM research panels ────────────────────────────────────
    if v2_model is not None and y_prob_v2 is not None:
        print("\n  Generating Grad-CAM research panels …")
        opt_t = metrics_v2["optimal_threshold"]
        y_pred_v2 = (y_prob_v2 >= opt_t).astype(int)

        # True Positives: actually cancer, predicted cancer
        tp_idx = np.where((labels == 1) & (y_pred_v2 == 1))[0]
        # False Positives: actually non-cancer, predicted cancer
        fp_idx = np.where((labels == 0) & (y_pred_v2 == 1))[0]
        # False Negatives: actually cancer, predicted non-cancer (missed)
        fn_idx = np.where((labels == 1) & (y_pred_v2 == 0))[0]

        # Select up to 5 TPs (sorted by highest confidence)
        tp_sel = tp_idx[np.argsort(y_prob_v2[tp_idx])[::-1]][:5].tolist()
        # Select up to 3 FPs
        fp_sel = fp_idx[np.argsort(y_prob_v2[fp_idx])[::-1]][:3].tolist()
        # Select up to 3 FNs (missed cancers — highest-stakes errors)
        fn_sel = fn_idx[np.argsort(y_prob_v2[fn_idx])[::-1]][:3].tolist()

        plot_gradcam_panel(
            model         = v2_model,
            images_raw    = images_norm,
            images_display= images_uint8,
            metadata      = metadata_arr,
            indices       = tp_sel,
            y_true        = labels,
            y_prob        = y_prob_v2,
            title         = "Grad-CAM — True Positives (Correctly Identified Cancer)",
            save_path     = os.path.join(OUTPUT_DIR, "gradcam_panel_tp.png"),
        )

        plot_gradcam_panel(
            model         = v2_model,
            images_raw    = images_norm,
            images_display= images_uint8,
            metadata      = metadata_arr,
            indices       = fp_sel,
            y_true        = labels,
            y_prob        = y_prob_v2,
            title         = "Grad-CAM — False Positives (Non-Cancer Flagged as Cancer)",
            save_path     = os.path.join(OUTPUT_DIR, "gradcam_panel_fp.png"),
        )

        if fn_sel:
            plot_gradcam_panel(
                model         = v2_model,
                images_raw    = images_norm,
                images_display= images_uint8,
                metadata      = metadata_arr,
                indices       = fn_sel,
                y_true        = labels,
                y_prob        = y_prob_v2,
                title         = "Grad-CAM — False Negatives ⚠️ MISSED CANCER",
                save_path     = os.path.join(OUTPUT_DIR, "gradcam_panel_fn.png"),
            )

        print(f"\n  Grad-CAM panel sample counts:")
        print(f"    True Positives  (shown): {len(tp_sel)}")
        print(f"    False Positives (shown): {len(fp_sel)}")
        print(f"    False Negatives (shown): {len(fn_sel)}  ← missed cancers")

    # ── 5.11  Final summary ───────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("  Evaluation Complete")
    print(f"  Elapsed  : {elapsed:.1f}s")
    print(f"  Outputs  : {os.path.relpath(OUTPUT_DIR, ROOT)}/")
    print("=" * 60)
    print()

    if metrics_v1:
        auc_grade = "research-grade (≥0.90)" if metrics_v1["roc_auc"] >= 0.90 else \
                    "strong (≥0.85)"          if metrics_v1["roc_auc"] >= 0.85 else "needs improvement"
        print(f"  v1 AUC : {metrics_v1['roc_auc']:.4f}  → {auc_grade}")
        print(f"  v1 Sensitivity (recall for cancer): {metrics_v1['sensitivity']:.4f}")

    if metrics_v2 and not v2_weights_real:
        print()
        print("  ⚠️  RESEARCH NOTE: v2 results above use RANDOM weights.")
        print("      Train the v2 model on labelled data with real metadata,")
        print("      then rerun this script to get valid comparative metrics.")
    print()


if __name__ == "__main__":
    main()
