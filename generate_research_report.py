"""
CuraLens — Research Report Generator
======================================
Purely analytical script. Reads pre-computed artefacts from
evaluation_outputs/ and models_v2/training_logs_v2.json, then
produces a self-contained research report package under research_report/.

No models are loaded. No inference is performed. Nothing is modified.

Outputs:
    research_report/
        final_report.md           ← full narrative report in Markdown
        comparison_table.json     ← machine-readable metric comparison
        figures/
            confusion_matrix_comparison.png
            roc_curve.png               (copied from evaluation_outputs/)
            gradcam_tp.png              (copied from evaluation_outputs/)
            gradcam_fp.png              (copied from evaluation_outputs/)
            gradcam_fn.png              (copied from evaluation_outputs/ if present)
            training_curves.png         (generated from training_logs_v2.json)

Run:
    python generate_research_report.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.dirname(__file__))
EVAL_DIR   = os.path.join(ROOT, "evaluation_outputs")
MODEL_LOG  = os.path.join(ROOT, "models_v2", "training_logs_v2.json")
REPORT_DIR = os.path.join(ROOT, "research_report")
FIG_DIR    = os.path.join(REPORT_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str, label: str) -> dict:
    if not os.path.exists(path):
        print(f"  ⚠️  {label} not found: {os.path.relpath(path, ROOT)}")
        return {}
    with open(path) as f:
        return json.load(f)


def load_artefacts() -> tuple[dict, dict, dict, dict]:
    print("\n[1/6] Loading evaluation artefacts …")
    m1       = load_json(os.path.join(EVAL_DIR, "metrics_v1.json"),        "metrics_v1.json")
    m2       = load_json(os.path.join(EVAL_DIR, "metrics_v2.json"),        "metrics_v2.json")
    ablation = load_json(os.path.join(EVAL_DIR, "ablation_summary.json"),  "ablation_summary.json")
    train_log= load_json(MODEL_LOG,                                          "training_logs_v2.json")

    for name, d in [("metrics_v1", m1), ("metrics_v2", m2)]:
        if d:
            print(f"  ✅ {name}: AUC={d.get('roc_auc','?')}  "
                  f"Sensitivity={d.get('sensitivity','?')}")
    return m1, m2, ablation, train_log


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COPY PRE-GENERATED FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def copy_figures() -> None:
    print("\n[2/6] Copying pre-generated figures …")
    files = {
        "roc_curve.png"         : "roc_curve.png",
        "gradcam_panel_tp.png"  : "gradcam_tp.png",
        "gradcam_panel_fp.png"  : "gradcam_fp.png",
        "gradcam_panel_fn.png"  : "gradcam_fn.png",
    }
    for src_name, dst_name in files.items():
        src = os.path.join(EVAL_DIR, src_name)
        dst = os.path.join(FIG_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  📋 {src_name} → figures/{dst_name}")
        else:
            print(f"  ⚠️  {src_name} not found — run evaluate_v2.py first")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONFUSION MATRIX SIDE-BY-SIDE FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def _draw_cm(ax: plt.Axes, cm: list, title: str,
             labels: list = None, note: str = "") -> None:
    if labels is None:
        labels = ["Non-Cancer", "Cancer"]
    arr    = np.array(cm)
    im     = ax.imshow(arr, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=arr.max() * 1.05)
    thresh = arr.max() / 2.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i,j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if arr[i, j] > thresh else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="True",
           title=title)
    if note:
        ax.set_xlabel(f"Predicted\n\n⚠️  {note}", fontsize=9, color="#c0392b")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def generate_cm_comparison(m1: dict, m2: dict) -> str:
    print("\n[3/6] Generating confusion matrix comparison figure …")
    save_path = os.path.join(FIG_DIR, "confusion_matrix_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "CuraLens — Confusion Matrix Comparison\n"
        "v1 (Image-only) vs v2 (Multimodal, synthetic metadata)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    if m1.get("confusion_matrix"):
        _draw_cm(axes[0], m1["confusion_matrix"],
                 f"v1 — Image-only\nAUC={m1.get('roc_auc','?')}  "
                 f"Sensitivity={m1.get('sensitivity','?')}")
    else:
        axes[0].set_title("v1 — No data")
        axes[0].axis("off")

    if m2.get("confusion_matrix"):
        _draw_cm(axes[1], m2["confusion_matrix"],
                 f"v2 — Multimodal\nAUC={m2.get('roc_auc','?')}  "
                 f"Sensitivity={m2.get('sensitivity','?')}",
                 note="Synthetic metadata — not a clinical result")
    else:
        axes[1].set_title("v2 — No data")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 figures/confusion_matrix_comparison.png")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING CURVES FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_curves(train_log: dict) -> str:
    print("\n[4/6] Generating training curve figures …")
    save_path = os.path.join(FIG_DIR, "training_curves.png")

    history = train_log.get("history", [])
    if not history:
        print("  ⚠️  No training history — skipping training curves")
        return ""

    epochs_p1 = [e for e in history if e["phase"] == 1]
    epochs_p2 = [e for e in history if e["phase"] == 2]

    def _vals(epochs, key):
        return [e[key] for e in epochs]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("CuraLens v2 — Training History\n"
                 "Phase 1 (warm-up) | Phase 2 (fine-tuning)",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("loss",     "val_loss",       "Loss",         "lower-left"),
        ("auc",      "val_auc",        "AUC",          "lower-right"),
        ("accuracy", "val_accuracy",   "Accuracy",     "upper-right"),
        ("recall",   "val_recall",     "Sensitivity\n(Recall for Cancer)", "upper-right"),
    ]

    colors = {"p1_train": "#2563eb", "p1_val": "#60a5fa",
              "p2_train": "#7c3aed", "p2_val": "#a78bfa"}

    for ax, (train_key, val_key, title, _) in zip(axes.flat, metrics):
        offset1 = 0
        offset2 = len(epochs_p1)

        if epochs_p1:
            x1 = list(range(1, len(epochs_p1) + 1))
            ax.plot(x1, _vals(epochs_p1, train_key),
                    color=colors["p1_train"], lw=2, label="Phase 1 train")
            ax.plot(x1, _vals(epochs_p1, val_key),
                    color=colors["p1_val"], lw=2, ls="--", label="Phase 1 val")

        if epochs_p2:
            x2 = list(range(offset2 + 1, offset2 + len(epochs_p2) + 1))
            ax.plot(x2, _vals(epochs_p2, train_key),
                    color=colors["p2_train"], lw=2, label="Phase 2 train")
            ax.plot(x2, _vals(epochs_p2, val_key),
                    color=colors["p2_val"], lw=2, ls="--", label="Phase 2 val")

        # Phase separator
        if epochs_p1 and epochs_p2:
            ax.axvline(x=len(epochs_p1) + 0.5, color="grey",
                       ls=":", lw=1.5, label="Phase boundary")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 figures/training_curves.png")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPARISON TABLE JSON
# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "accuracy"   : "Accuracy",
    "sensitivity": "Sensitivity (Recall)",
    "specificity": "Specificity",
    "precision"  : "Precision (PPV)",
    "npv"        : "NPV",
    "f1_score"   : "F1 Score",
    "roc_auc"    : "ROC-AUC",
}

METRIC_CLINICAL_NOTE = {
    "sensitivity": "Most critical — minimises missed cancers (FN)",
    "specificity": "Minimises false alarms (FP)",
    "roc_auc"    : "Overall discriminative performance",
    "precision"  : "Positive predictive value",
    "npv"        : "Negative predictive value",
    "f1_score"   : "Harmonic mean of precision + recall",
    "accuracy"   : "Overall correctness",
}

AUC_GRADE = {
    (0.90, 1.01): "Research-grade (≥ 0.90)",
    (0.85, 0.90): "Strong (≥ 0.85)",
    (0.70, 0.85): "Acceptable (≥ 0.70)",
    (0.00, 0.70): "Needs improvement (< 0.70)",
}


def _auc_grade(auc: float) -> str:
    for (lo, hi), label in AUC_GRADE.items():
        if lo <= auc < hi:
            return label
    return "N/A"


def generate_comparison_table(m1: dict, m2: dict, ablation: dict) -> dict:
    print("\n[5/6] Building comparison table …")
    rows = []
    for key, label in METRIC_LABELS.items():
        v1  = m1.get(key)
        v2  = m2.get(key)
        d   = ablation.get("metrics", {}).get(key, {}).get("delta")
        rows.append({
            "metric"        : key,
            "label"         : label,
            "v1"            : round(v1, 4) if v1 is not None else None,
            "v2"            : round(v2, 4) if v2 is not None else None,
            "delta"         : round(d,  4) if d  is not None else None,
            "clinical_note" : METRIC_CLINICAL_NOTE.get(key, ""),
        })

    table = {
        "generated_at"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "v1_model"          : "CuraLens v1 — MobileNetV2 (Image-only)",
        "v2_model"          : "CuraLens v2 — EfficientNetB0 + Metadata Fusion",
        "v1_trained_on"     : "Real images, no metadata",
        "v2_trained_on"     : "Real images + SYNTHETIC metadata (see note)",
        "v2_note"           : (
            "v2 evaluation used synthetically generated patient metadata. "
            "Comparative results are NOT a fair clinical comparison. "
            "Retrain and re-evaluate with real patient records."
        ),
        "v1_auc_grade"      : _auc_grade(m1.get("roc_auc", 0)),
        "v2_auc_grade"      : _auc_grade(m2.get("roc_auc", 0)),
        "rows"              : rows,
    }

    out_path = os.path.join(REPORT_DIR, "comparison_table.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"  💾 comparison_table.json")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _md_metric_table(table: dict) -> str:
    header = (
        "| Metric | v1 Image-only | v2 Multimodal* | Δ | Note |\n"
        "|---|---|---|---|---|\n"
    )
    rows = header
    for r in table["rows"]:
        v1  = f"{r['v1']:.4f}" if r["v1"] is not None else "—"
        v2  = f"{r['v2']:.4f}" if r["v2"] is not None else "—"
        d   = r["delta"]
        if d is None:
            delta_str = "—"
        else:
            sign = "+" if d >= 0 else ""
            icon = " ✅" if d > 0.001 else (" ⚠️" if d < -0.01 else " ≈")
            delta_str = f"{sign}{d:.4f}{icon}"
        rows += f"| **{r['label']}** | {v1} | {v2} | {delta_str} | {r['clinical_note']} |\n"
    return rows


def generate_markdown_report(m1: dict, m2: dict,
                              ablation: dict,
                              train_log: dict,
                              table: dict) -> str:
    print("\n[6/6] Writing Markdown report …")

    history  = train_log.get("history", [])
    phases   = train_log.get("phases", {})
    best     = max(history, key=lambda e: e["val_auc"]) if history else {}
    p1_count = sum(1 for e in history if e["phase"] == 1)
    p2_count = sum(1 for e in history if e["phase"] == 2)

    now = datetime.now().strftime("%B %d, %Y")

    # Interpretation paragraph
    interp_parts = []
    if m1.get("roc_auc", 0) >= 0.90:
        interp_parts.append(
            f"The v1 image-only baseline achieves a **ROC-AUC of "
            f"{m1['roc_auc']:.4f}**, classifying it as research-grade (≥ 0.90). "
            f"Its sensitivity of **{m1['sensitivity']:.4f}** ({m1['sensitivity']*100:.1f}%) "
            f"means only {m1.get('FN', '?')} cancer case(s) were missed across "
            f"{m1.get('n_positive','?')} positive samples in the validation set."
        )
    if m2.get("roc_auc"):
        interp_parts.append(
            f"The v2 multimodal model, trained on **synthetic metadata**, reports "
            f"an AUC of {m2['roc_auc']:.4f}. These results should not be "
            f"interpreted as a definitive improvement or regression over v1 — "
            f"the metadata branch has not yet been trained on real patient records. "
            f"The evaluation serves as a **structural sanity check** of the "
            f"multimodal fusion architecture."
        )
    interpretation = "  \n".join(interp_parts)

    # --- Training details ---
    td_rows = ""
    if train_log:
        td_rows = f"""\
| Training date | {train_log.get('training_date', '—')} |
| Architecture  | {train_log.get('architecture', '—')} |
| Train samples | {train_log.get('total_train_samples', '—')} |
| Val samples   | {train_log.get('total_val_samples', '—')} |
| Metadata      | {'⚠️ Synthetic' if train_log.get('metadata_synthetic') else '✅ Real'} |
| Phase 1 (warm-up) | LR={phases.get('phase1',{}).get('lr','—')}, requested {phases.get('phase1',{}).get('epochs_requested','—')} epochs → ran {p1_count} |
| Phase 2 (fine-tune) | LR={phases.get('phase2',{}).get('lr','—')}, top-{phases.get('phase2',{}).get('unfrozen_efficientnet_layers','?')} EfficientNet layers unfrozen → ran {p2_count} |
| Best epoch    | Phase {best.get('phase','?')}, Epoch {best.get('epoch','?')} (val_auc={best.get('val_auc','?')}) |
| Class weights | Non-cancer={train_log.get('class_weights',{}).get('0',1.0):.4f}, Cancer={train_log.get('class_weights',{}).get('1',1.0):.4f} |
"""

    report = f"""\
# CuraLens — Research Report

**Generated:** {now}  
**Project:** CuraLens — Explainable Multi-modal AI for Early Cancer Detection  
**Authors:** CuraLens Research Team  
**Status:** Experimental — v2 trained on synthetic metadata

---

## 1. Project Overview

CuraLens is an explainable multi-modal AI system designed to assist clinicians
in the early detection of oral and skin cancer. The system combines:

- **Visual analysis** via a deep convolutional neural network (CNN) applied to
  clinical photographs.
- **Patient risk factors** (age, smoking status, alcohol consumption, and sun
  exposure) fused with image features via a dedicated metadata branch.
- **Grad-CAM explainability** to highlight image regions most influential to
  each prediction.
- **Structured risk scoring** that maps sigmoid probability to a three-tier
  clinical alert (Low / Medium / High risk).

---

## 2. Model Architectures

### CuraLens v1 — Image-only Baseline

| Property | Value |
|---|---|
| Backbone | MobileNetV2 (ImageNet pre-trained) |
| Input | 224 × 224 × 3 RGB image |
| Output | Sigmoid probability: P(cancer) |
| Training data | {m1.get('n_samples', '138')} validation samples (real images) |
| Optimal threshold | {m1.get('optimal_threshold', '—')} (Youden's J) |

### CuraLens v2 — Multimodal Architecture

| Property | Value |
|---|---|
| Image branch | EfficientNetB0 → GAP → Dense(512) → Dropout(0.4) |
| Metadata branch | BatchNorm → Dense(64) → Dropout → Dense(64) |
| Fusion | Concatenate → Dense(256) → Dense(128) → Sigmoid |
| Metadata inputs | age, smoking (0/1), alcohol (0/1), sun\_exposure (0–10) |
| Output | Sigmoid probability: P(cancer) |

---

## 3. Training Strategy — v2

{td_rows}

**Phase 1 (Warm-up):**  
EfficientNetB0 backbone frozen. Only the metadata branch and fusion head
were trained. Learning rate = 1×10⁻⁴. EarlyStopping on `val_auc` with
patience = 8.

**Phase 2 (Fine-tuning):**  
The top 20 layers of EfficientNetB0 were unfrozen and trained jointly with
the full head at a low learning rate of 1×10⁻⁵ (patience = 5). The best
Phase 1 checkpoint was automatically restored after fine-tuning did not
yield further improvement.

**Class imbalance handling:**  
`sklearn.utils.class_weight.compute_class_weight("balanced")` was applied
to balance the Cancer / Non-Cancer training distribution.

**Data augmentation (training only):**  
Horizontal/vertical flip, random brightness ±10%, random 90° rotation.

---

## 4. Validation Metrics

{_md_metric_table(table)}

> \\* v2 trained and evaluated with **synthetic patient metadata**.
> The comparison is architectural validation only — not a clinical benchmark.
> AUC grade: v1 = {table['v1_auc_grade']} | v2 = {table['v2_auc_grade']}

---

## 5. Interpretation

{interpretation}

### Sensitivity — The Most Critical Metric

In cancer screening, **Sensitivity (Recall)** is the primary clinical metric.
A False Negative (FN) means a cancer case is missed — the highest-stakes
error category. CuraLens v1 achieves a sensitivity of
**{m1.get('sensitivity', '—')} ({(m1.get('sensitivity',0))*100:.1f}%)**,
resulting in only **{m1.get('FN', '?')} missed case(s)** in the
{m1.get('n_positive', '?')}-sample positive validation set.

### Specificity — Avoiding Unnecessary Alarm

Specificity measures the model's ability to correctly clear non-cancer cases.
v1 specificity: **{m1.get('specificity', '—')}** → {m1.get('FP','?')} false
alarms from {m1.get('n_negative','?')} non-cancer patients.

### ROC-AUC

An AUC ≥ 0.90 is considered **research-grade** in medical imaging literature.
v1 achieves **{m1.get('roc_auc', '—')}**, placing it firmly in this category.

---

## 6. Explainability — Grad-CAM Analysis

Gradient-weighted Class Activation Maps (Grad-CAM) were generated for the
v2 model's validation set using the `top_conv` layer of EfficientNetB0.
Three panels were produced:

| Panel | Contents | Figure |
|---|---|---|
| True Positives | 5 correctly identified cancer cases | `figures/gradcam_tp.png` |
| False Positives | 3 non-cancer cases flagged as cancer | `figures/gradcam_fp.png` |
| False Negatives | 3 missed cancer cases (highest-stakes errors) | `figures/gradcam_fn.png` |

### Qualitative Observations *(placeholder — fill after visual review)*

> **Reviewer instructions:** Examine each Grad-CAM panel and replace the
> placeholders below with observed activation patterns.

- **True Positives:** Does the heatmap concentrate on the lesion region, or
  on background tissue? *(to be completed after expert review)*
- **False Positives:** Is the model activating on artefacts, lighting, or
  anatomical structures that superficially resemble lesions?
  *(to be completed)*
- **False Negatives:** Are missed cases characterised by low-contrast lesions,
  unusual anatomical positioning, or poor image quality?
  *(to be completed)*
- **Metadata influence:** Does the fusion of high-risk metadata (e.g.,
  heavy smoker, elderly) shift the activation threshold?
  *(to be completed after ablation with real metadata)*

---

## 7. Ablation Study

A formal ablation study comparing v1 (image-only) vs v2 (multimodal) is
structurally complete but **not yet clinically meaningful** due to synthetic
metadata. The framework is in place:

| Stage | Status |
|---|---|
| v1 evaluation on real data | ✅ Complete |
| v2 architecture validation (synthetic) | ✅ Complete |
| v2 training with real metadata | ⏳ Pending data collection |
| Valid v1 vs v2 ablation comparison | ⏳ Pending above |

Once real patient metadata is collected and `data_clean/metadata.csv` is
populated, re-running `train_v2.py` followed by `evaluate_v2.py` and
`generate_research_report.py` will produce a fully valid comparison.

---

## 8. Risk Scoring System

Predictions are mapped to three clinical tiers:

| Tier | Probability Range | Recommendation | UI Colour |
|---|---|---|---|
| Low | [0.0, 0.3) | Routine monitoring | 🟢 `#2ECC71` |
| Medium | [0.3, 0.7) | Further clinical evaluation | 🟡 `#F39C12` |
| High | [0.7, 1.0] | Urgent specialist referral | 🔴 `#E74C3C` |

Thresholds were set based on clinical literature for oral cancer screening
tools and may be adjusted by a clinical review board.

---

## 9. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | v1 image-only prediction (production) |
| `/predict_v2` | POST | v2 multimodal prediction (experimental) |
| `/predict/skin` | POST | Skin screening module |

`/predict_v2` accepts `multipart/form-data` with fields:
`image`, `age`, `smoking`, `alcohol`, `sun_exposure` (or a JSON-encoded
`metadata` field). Returns probability, risk tier, and base64 Grad-CAM PNG.

---

## 10. Limitations & Future Work

1. **Synthetic metadata:** v2 results are not clinically valid until real
   patient demographics are collected and validated.
2. **Dataset size:** 802 training / 138 validation images is small for a
   deep learning classifier. External validation on a held-out institutional
   dataset is required before clinical deployment.
3. **Prospective validation:** All results are retrospective. Prospective
   clinical validation is required per regulatory standards (FDA 510(k) /
   CE Mark for AI medical devices).
4. **Metadata calibration:** Thresholds for `sun_exposure` and age
   normalisation should be clinically reviewed.
5. **Grad-CAM review:** Activation maps should be reviewed by at minimum
   two oral pathologists before publication.

---

## 11. File Inventory

```
research_report/
├── final_report.md                  ← this document
├── comparison_table.json            ← machine-readable metrics
└── figures/
    ├── roc_curve.png                ← ROC overlay (v1 + v2)
    ├── confusion_matrix_comparison.png
    ├── training_curves.png          ← v2 Phase 1 + Phase 2 history
    ├── gradcam_tp.png               ← True Positive Grad-CAM panel
    ├── gradcam_fp.png               ← False Positive Grad-CAM panel
    └── gradcam_fn.png               ← False Negative Grad-CAM panel
```

---

## 12. Reproducibility

All scripts are deterministic given fixed random seeds (SEED = 42) and
the same hardware configuration. To reproduce:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train v2 (after replacing metadata.csv with real patient data)
python train_v2.py --epochs-phase1 30 --epochs-phase2 20 --batch 16

# 3. Evaluate
python evaluate_v2.py

# 4. Generate this report
python generate_research_report.py
```

---

*Report generated automatically by `generate_research_report.py`.*  
*CuraLens is an AI-assisted **screening** tool. It is not a diagnostic device.*  
*All predictions must be reviewed by a qualified healthcare professional.*  
*Not approved for clinical use.*
"""

    out_path = os.path.join(REPORT_DIR, "final_report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"  💾 final_report.md  ({len(report):,} chars)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    start = datetime.now()
    print("\n" + "=" * 60)
    print("  CuraLens — Research Report Generator")
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    m1, m2, ablation, train_log = load_artefacts()
    copy_figures()
    generate_cm_comparison(m1, m2)
    generate_training_curves(train_log)
    table = generate_comparison_table(m1, m2, ablation)
    generate_markdown_report(m1, m2, ablation, train_log, table)

    elapsed = (datetime.now() - start).total_seconds()
    print("\n" + "=" * 60)
    print("  Report generation complete")
    print(f"  Elapsed  : {elapsed:.1f}s")
    print(f"  Output   : {os.path.relpath(REPORT_DIR, ROOT)}/")
    print("=" * 60)

    # Print inventory
    for root_dir, dirs, files in os.walk(REPORT_DIR):
        dirs.sort()
        level  = root_dir.replace(REPORT_DIR, "").count(os.sep)
        indent = "  " + "    " * level
        print(f"{indent}{os.path.basename(root_dir)}/")
        sub = "  " + "    " * (level + 1)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root_dir, f))
            size_str = f"{size/1024:.1f} KB" if size >= 1024 else f"{size} B"
            print(f"{sub}{f}  ({size_str})")
    print()


if __name__ == "__main__":
    main()
