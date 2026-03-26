# CuraLens 🩺  
### AI-Assisted Multi-Module Medical Screening Platform

CuraLens is a deep learning–based web platform designed to assist in the **screening of oral cavity and skin images** for **abnormal patterns**.  
It functions as an **AI-assisted decision-support system** and **does not provide medical diagnosis** or replace clinical judgment.

---

## 🔍 Project Motivation
Oral cancer and skin malignancies have a high prevalence, particularly in countries like India.  
Early-stage **screening and risk flagging** can help guide individuals toward timely clinical evaluation.

This project explores how **computer vision, transfer learning, and clinical metadata fusion** can support **preliminary screening** in an ethical and responsible manner.

---

## 🚀 Key Features
- **Dual-modality screening**: Oral cavity (v1 + v2/v3) and skin lesion (v1 + v3)
- **Multimodal fusion**: EfficientNetB0 image branch fused with 6D clinical metadata
- **Grad-CAM explainability** on both v1 and v2/v3 models — visual heatmaps highlight suspicious regions
- **Three-tier risk scoring**: Low / Medium / High with colour codes and clinical recommendations
- **Metadata schema validation** with graceful degradation on missing or out-of-range fields
- **Two-phase training strategy**: warm-up (CNN frozen) → fine-tuning (top-20 EfficientNet layers unfrozen)
- **Focal loss support** for handling class imbalance
- **Stratified K-fold cross-validation** with saved per-fold metrics
- **REST API** with `/predict`, `/predict/skin`, `/predict_v2`, `/schema/<type>`, `/health`
- **Flask SPA web interface** with animated gradients and real-time Grad-CAM display
- **Automatic prediction logging** to `automation_logs/`
- **System health endpoint** for monitoring model load status

---

## 🧠 Model Architecture Overview

### v1 — Image-Only Models (MobileNetV2)
| Model | Architecture | Val AUC | Input |
|---|---|---|---|
| Oral v1 | MobileNetV2 (frozen) → Dense(128) → Sigmoid | **0.993** | 224×224 RGB |
| Skin v1 | MobileNetV2 (frozen) → Dense(128) → Sigmoid | **0.943** | 224×224 RGB |

### v2 / v3 — Multimodal Models (EfficientNetB0 + Metadata)
```
Image Input (224×224×3)          Metadata Input (6D clinical)
        ↓                                    ↓
EfficientNetB0 (frozen/fine-tuned)     BatchNorm
GlobalAvgPool → Dense(512) → Dropout   Dense(64) → Dense(64)
        ↓                                    ↓
        └────────── Concatenate (576D) ──────┘
                          ↓
                  Dense(256) → Dense(128) → Sigmoid → P(cancer)
```

**Oral v3 metadata (6D):** age, smoking_years, cigarettes_per_day, alcohol_units_per_week, chewing_tobacco, family_history  
**Skin v3 metadata (6D):** age, skin_type (Fitzpatrick 1-6), sunburn_history, outdoor_hours_per_week, tanning_bed_use, family_history

---

## 🌐 API Reference

### `POST /predict` — Oral v1 (Image-Only)
```bash
curl -X POST http://localhost:5001/predict \
  -F "image=@path/to/oral.jpg" \
  -F "mode=diagnostic"
```
Response includes `cancer_probability`, `risk_level`, `recommendation`, `gradcam_png_b64`.

### `POST /predict/skin` — Skin v1 (Image-Only)
```bash
curl -X POST http://localhost:5001/predict/skin \
  -F "image=@path/to/lesion.jpg"
```

### `POST /predict_v2` — Multimodal v2/v3
```bash
# Oral v3 with clinical metadata
curl -X POST http://localhost:5001/predict_v2 \
  -F "cancer_type=oral" \
  -F "image=@oral.jpg" \
  -F "age=52" \
  -F "smoking_years=15" \
  -F "cigarettes_per_day=10" \
  -F "alcohol_units_per_week=7" \
  -F "chewing_tobacco=0" \
  -F "family_history=1"

# Skin v3 with clinical metadata
curl -X POST http://localhost:5001/predict_v2 \
  -F "cancer_type=skin" \
  -F "image=@lesion.jpg" \
  -F "age=45" \
  -F "skin_type=2" \
  -F "sunburn_history=8" \
  -F "outdoor_hours_per_week=20" \
  -F "tanning_bed_use=0" \
  -F "family_history=0"
```
Response includes `probability`, `risk_level`, `risk_label`, `confidence_band`, `recommendation`, `color_code`, `gradcam_png_b64`.

### `GET /schema/<cancer_type>` — Metadata Field Schema
```bash
curl http://localhost:5001/schema/oral
curl http://localhost:5001/schema/skin
curl http://localhost:5001/schema/oral_legacy
```

### `GET /health` — System Health Check
```bash
curl http://localhost:5001/health
```
Returns load status of all 5 model variants, available endpoints, and overall `"ok"` / `"degraded"` status.

---

## 📂 Project Structure

```
OralCancerApp/
├── train.py                  # v1 oral training (MobileNetV2)
├── train_v2.py               # v2/v3 multimodal training pipeline
├── train_skin.py             # v1 skin training (MobileNetV2)
├── predict.py                # CLI prediction (v1 oral)
├── evaluate_v2.py            # v1 vs v2 ablation evaluation
├── evaluate_skin.py          # Skin model test-set evaluation ← NEW
├── generate_research_report.py  # Auto-generate research report
├── web_app.py                # Flask web application + REST API
├── system_smoke_test.py      # End-to-end HTTP integration tests
├── requirements.txt          # Python dependencies
│
├── models/                   # v1 trained models
│   ├── oral_cancer_model.h5
│   ├── model_metadata.json
│   └── skin_model/
│       └── skin_screening_model.h5
│
├── models_v2/                # v2/v3 architectures + trained weights
│   ├── multimodal_model.py   # Legacy oral 4D architecture
│   ├── oral_model.py         # Oral v3 6D architecture
│   ├── skin_model.py         # Skin v3 6D architecture
│   ├── metadata_scaler.pkl   # Fitted StandardScaler (saved after training)
│   ├── saved_model/          # oral_legacy SavedModel
│   ├── oral_saved_model/     # oral v3 SavedModel
│   └── skin_saved_model/     # skin v3 SavedModel
│
├── utils_v2/
│   ├── gradcam.py            # Grad-CAM explainability (single + multi-input)
│   ├── metadata_schema.py    # Field definitions, validation, normalisation
│   └── risk_scoring.py       # Risk tier logic (Low/Medium/High)
│
├── modules/
│   └── skin_screening.py     # v1 skin screening wrapper
│
├── data_clean/               # Training data (oral)
│   ├── metadata.csv
│   └── train/ val/
│
├── skin_dataset_resized/     # Training data (skin)
│   └── train_set/ val_set/ test_set/
│
├── evaluation_outputs/       # Metrics, ROC curves, confusion matrices
├── research_report/          # Auto-generated publication report
├── automation_logs/          # Prediction history JSONs
└── test_assets/              # Test images for smoke tests
    └── sample.jpg
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Web Application
```bash
python web_app.py
# or specify a port:
python web_app.py 8080
```
Open: http://localhost:5001

### 3️⃣ Train Models

**v1 oral model (MobileNetV2):**
```bash
python train.py
```

**v2 multimodal oral model (EfficientNetB0 + 4D metadata):**
```bash
python train_v2.py --epochs-phase1 30 --epochs-phase2 20
```

**v3 skin multimodal model (EfficientNetB0 + 6D metadata):**
```bash
python train_v2.py --cancer-type skin --epochs-phase1 30 --epochs-phase2 20
```

**With focal loss (recommended when dataset is imbalanced):**
```bash
python train_v2.py --cancer-type oral --use-focal-loss
```

**With cross-validation:**
```bash
python train_v2.py --cross-validate --cv-folds 5
```

### 4️⃣ Evaluate Models

**v1 vs v2 oral ablation evaluation:**
```bash
python evaluate_v2.py
```

**Skin model on held-out test set:**
```bash
python evaluate_skin.py
```

**Generate research report:**
```bash
python generate_research_report.py
```

### 5️⃣ Run Smoke Tests
```bash
# Start the server first, then in another terminal:
python system_smoke_test.py

# Or auto-start the server:
python system_smoke_test.py --autostart
```

### 6️⃣ CLI Prediction (v1 oral)
```bash
python predict.py path/to/image.jpg
python predict.py image.jpg 0.35   # custom threshold
```

---

## 🎯 Risk Scoring System

| Tier | P(cancer) Range | Color | Action |
|---|---|---|---|
| Low | 0.0 – 0.3 | 🟢 Green | Routine monitoring |
| Medium | 0.3 – 0.7 | 🟡 Amber | Further clinical evaluation |
| High | 0.7 – 1.0 | 🔴 Red | Urgent specialist referral |

---

## 📈 Model Performance

| Model | Val AUC | Sensitivity | Specificity | Notes |
|---|---|---|---|---|
| Oral v1 (MobileNetV2) | **0.993** | **0.986** | **0.955** | Real images, image-only |
| Skin v1 (MobileNetV2) | 0.943 | — | — | Real images, image-only |
| Oral v2 (EfficientNetB0 + 4D) | 0.784 | 0.845 | 0.612 | ⚠️ Synthetic metadata |
| Oral v2 training peak | (**0.992**) | — | — | Synthetic metadata |

> **Note:** v2/v3 multimodal metrics reflect synthetic metadata. Replace `data_clean/metadata.csv` with real patient records before any clinical or research claims.

---

## ⚠️ Disclaimer (Important)

This system is developed **strictly for educational and research purposes**.

- The model performs **image-based screening only**
- It does **not diagnose cancer or any disease**
- Results must **always be reviewed by qualified medical professionals**
- Clinical decisions **must not** be made based on this tool alone
- v2/v3 multimodal models are trained on **synthetic metadata** — not validated for clinical claims

---

## 🔮 Future Scope
- Collect real patient metadata to unlock full multimodal accuracy potential
- Train oral v3 (6D schema) — architecture is ready, training script needed
- Multi-class oral abnormality categorisation (leukoplakia, erythroplakia, etc.)
- REST API authentication and rate limiting for research integration
- Mobile application interface
- Docker containerisation for reproducible deployment

---

## 👨‍🎓 Author

**Jay Gautam**  
B.Tech – Computer Science (Artificial Intelligence & Machine Learning)

---

## 🟢 Project Status
| Component | Status |
|---|---|
| Oral screening v1 (MobileNetV2) | ✅ Complete |
| Skin screening v1 (MobileNetV2) | ✅ Complete |
| v2 Multimodal oral (EfficientNetB0 + 4D) | ✅ Trained (synthetic metadata) |
| v3 Skin multimodal (EfficientNetB0 + 6D) | ✅ Architecture ready |
| v3 Oral multimodal (EfficientNetB0 + 6D) | 🟡 Architecture ready, training pending |
| Grad-CAM explainability (v1 + v2/v3) | ✅ Complete |
| Flask REST API + SPA UI | ✅ Complete |
| Research report pipeline | ✅ Complete |
| Real patient metadata | ❌ Pending (synthetic used for now) |
| Clinical validation | ❌ Out of scope |

---

> *CuraLens is a technical exploration of AI-assisted screening, designed with responsibility, transparency, and academic integrity at its core.*

