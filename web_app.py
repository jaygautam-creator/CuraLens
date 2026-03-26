from flask import Flask, request, jsonify, render_template_string, send_file
import os
import io
import base64
import uuid
import numpy as np
import json
from datetime import datetime
import traceback
import sys
import os

# Prioritize our isolated dependencies from /tmp/cura_lib
if os.path.exists('/tmp/cura_lib'):
    sys.path.insert(0, '/tmp/cura_lib')

# Workaround for BatchNormalization deserialization error (axis list vs int)
# and centralized model loading logic.
from utils_v2.model_loader import safe_load_model

try:
    from modules.skin_screening import predict_skin
    SKIN_MODULE_AVAILABLE = True
except ImportError as e:
    SKIN_MODULE_AVAILABLE = False
    print(f"⚠️ Skin screening module not available: {e}")

try:
    import tensorflow as tf
    import cv2
    from PIL import Image
    ORAL_DEPENDENCIES_OK = True
except ImportError as e:
    ORAL_DEPENDENCIES_OK = False
    print(f"❌ Missing dependency for oral screening: {e}")

# ---------------------------------------------------------------------------
# v2 multi-modal module imports (isolated; never touches v1 model files)
# ---------------------------------------------------------------------------
try:
    import base64
    import sys as _sys
    _sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from models_v2.multimodal_model import (
        build_multimodal_model as _build_v2,
        load_model as _load_v2_from_disk,
    )
    from models_v2.oral_model import load_oral_model as _load_oral_v3_from_disk
    from models_v2.skin_model import load_skin_model as _load_skin_v3_from_disk
    from utils_v2.risk_scoring import score_prediction as v2_score
    from utils_v2.gradcam import generate_gradcam as v2_gradcam
    from utils_v2.metadata_schema import (
        validate_and_encode as _validate_meta,
        get_schema_info as _get_schema_info,
    )
    V2_AVAILABLE = True
except ImportError as e:
    V2_AVAILABLE = False
    print(f"⚠️ CuraLens v2 modules not available: {e}")

app = Flask(__name__)
oral_model = None
metadata = None

# v2 model is loaded lazily on first request (keeps startup fast)
_v2_model      = None   # legacy oral (4D metadata) — models_v2/saved_model/
_oral_v3_model = None   # oral v3 (6D clinical)     — models_v2/oral_saved_model/
_skin_v3_model = None   # skin multimodal (6D)      — models_v2/skin_saved_model/

def load_oral_model_and_metadata():
    global oral_model, metadata
    try:
        model_path = 'models/oral_cancer_model.h5'
        if not os.path.exists(model_path):
            print(f"⚠️ {model_path} not found, trying best_model.h5...")
            model_path = 'models/best_model.h5'
        
        oral_model = safe_load_model(model_path, compile=False)
        print(f"✅ Oral model loaded from {model_path}")
        
        metadata_path = 'models/model_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded: AUC {metadata['performance']['auc']:.4f}")
        else:
            print("⚠️ Metadata not found, using defaults")
            metadata = {
                'performance': {
                    'auc': 0.9889,
                    'optimal_threshold': 0.512,
                    'sensitivity': 0.9296,
                    'specificity': 0.9701
                }
            }
        
        return True, "Oral model and metadata loaded"
    except Exception as e:
        return False, str(e)

def _load_v2_model():
    """
    Lazy-load the v2 multi-modal model.
    Tries to restore a saved model from models_v2/; falls back to building
    a fresh (untrained) model so the endpoint is always responsive.
    """
    global _v2_model
    if _v2_model is not None:
        return True, "v2 model already loaded"
    try:
        saved_path = os.path.join('models_v2', 'saved_model')
        if os.path.exists(saved_path):
            _v2_model = _load_v2_from_disk(saved_path)
            return True, f"v2 model restored from {saved_path}"
        # No saved weights yet — build architecture with random weights.
        # Replace this with _load_v2_from_disk() once training is complete.
        _v2_model = _build_v2(trainable_cnn=False)
        print("⚠️  [v2] No saved weights found. Using randomly-initialised model.")
        return True, "v2 model built (no saved weights — predictions are random)"
    except Exception as e:
        return False, str(e)


def _load_oral_v3_model():
    """Lazy-load the oral v3 model (6D clinical metadata)."""
    global _oral_v3_model
    if _oral_v3_model is not None:
        return True, "oral_v3 model already loaded"
    try:
        _oral_v3_model = _load_oral_v3_from_disk(fallback_legacy=True)
        return True, "oral_v3 model loaded"
    except Exception as e:
        return False, str(e)


def _load_skin_v3_model():
    """Lazy-load the skin multimodal v3 model (6D clinical metadata)."""
    global _skin_v3_model
    if _skin_v3_model is not None:
        return True, "skin_v3 model already loaded"
    try:
        _skin_v3_model = _load_skin_v3_from_disk()
        return True, "skin_v3 model loaded"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Patient record storage
# ---------------------------------------------------------------------------
PATIENTS_DIR = os.path.join(os.path.dirname(__file__), 'patients')
os.makedirs(PATIENTS_DIR, exist_ok=True)


def _save_patient_record(record: dict) -> str:
    """Persist a patient screening record to patients/<id>.json. Returns patient_id."""
    patient_id = record.get('patient_id') or str(uuid.uuid4())[:8].upper()
    record['patient_id'] = patient_id
    fpath = os.path.join(PATIENTS_DIR, f"{patient_id}.json")
    # Never store the raw image bytes or base64 in the record file — only metadata
    clean = {k: v for k, v in record.items() if k not in ('image_b64', 'gradcam_png_b64')}
    with open(fpath, 'w') as f:
        json.dump(clean, f, indent=2)
    return patient_id


def _load_patient_record(patient_id: str) -> dict | None:
    fpath = os.path.join(PATIENTS_DIR, f"{patient_id}.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)


def _list_patient_records(limit: int = 100) -> list:
    records = []
    try:
        files = sorted(
            [f for f in os.listdir(PATIENTS_DIR) if f.endswith('.json')],
            key=lambda f: os.path.getmtime(os.path.join(PATIENTS_DIR, f)),
            reverse=True
        )
        for fname in files[:limit]:
            with open(os.path.join(PATIENTS_DIR, fname)) as fh:
                r = json.load(fh)
                # Flatten fields for the frontend history table
                if 'patient_info' in r:
                    p = r['patient_info']
                    r['name'] = p.get('name', '—')
                    r['age'] = p.get('age', '—')
                    r['village'] = p.get('village', '—')
                if 'screening_date' in r:
                    r['screened_at'] = r['screening_date']
                records.append(r)
    except Exception:
        pass
    return records


# ---------------------------------------------------------------------------
# PDF report generator
# ---------------------------------------------------------------------------

def _build_pdf_report(record: dict,
                       image_bytes: bytes | None,
                       gradcam_bytes: bytes | None) -> io.BytesIO:
    """
    Build a one-page printable screening report PDF using ReportLab.
    Returns an in-memory BytesIO buffer containing the PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image as RLImage,
                                     HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.5 * cm, leftMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    W, H = A4  # 595 × 842 pt

    # ── custom styles ────────────────────────────────────────────────────────
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontSize=18, textColor=colors.HexColor('#1e3a8a'),
                                  spaceAfter=4, alignment=TA_CENTER)
    sub_style   = ParagraphStyle('Sub', parent=styles['Normal'],
                                  fontSize=9, textColor=colors.HexColor('#64748b'),
                                  alignment=TA_CENTER, spaceAfter=4)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'],
                                    fontSize=11, textColor=colors.HexColor('#1e3a8a'),
                                    spaceBefore=8, spaceAfter=4)
    body_style  = ParagraphStyle('Body', parent=styles['Normal'],
                                  fontSize=9, leading=13)
    risk_colors = {
        'HIGH':   ('#fef2f2', '#ef4444', '#991b1b'),
        'MEDIUM': ('#fffbeb', '#f59e0b', '#92400e'),
        'LOW':    ('#f0fdf4', '#22c55e', '#166534'),
    }

    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    story.append(Paragraph("CuraLens 🩺", title_style))
    story.append(Paragraph("AI-Assisted Medical Screening Report", sub_style))
    story.append(Paragraph(
        "⚠️  This report is for screening support only. Not a medical diagnosis. "
        "Always consult a qualified healthcare professional.", sub_style))
    story.append(HRFlowable(width='100%', thickness=1,
                              color=colors.HexColor('#e2e8f0'), spaceAfter=8))

    # ── Patient info table ───────────────────────────────────────────────────
    story.append(Paragraph("Patient Information", section_style))

    pi = record.get('patient_info', {})
    screening_type = record.get('screening_type', 'Oral').title()
    info_rows = [
        ['Patient ID',    record.get('patient_id', '—'),
         'Date',          record.get('screening_date', datetime.now().strftime('%Y-%m-%d %H:%M'))],
        ['Name',          pi.get('name', '—'),
         'Age / Gender',  f"{pi.get('age', '—')} yrs / {pi.get('gender', '—')}"],
        ['Village / Area', pi.get('village', '—'),
         'Contact',       pi.get('contact', '—')],
        ['Screening Type', screening_type,
         'Screened by',   pi.get('screened_by', '—')],
    ]
    tbl = Table(info_rows, colWidths=[(W - 3 * cm) / 4] * 4)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0f2fe')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#e0f2fe')),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('FONTNAME',   (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',   (2, 0), (2, -1), 'Helvetica-Bold'),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
         [colors.HexColor('#f8fafc'), colors.white]),
        ('PADDING',    (0, 0), (-1, -1), 5),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    # ── AI Result ────────────────────────────────────────────────────────────
    story.append(Paragraph("AI Screening Result", section_style))

    risk_level = record.get('risk_level', 'LOW').upper()
    raw_risk   = risk_level if risk_level in risk_colors else 'LOW'
    bg, border, text_c = risk_colors[raw_risk]
    prob_pct = record.get('probability_pct', 0)
    risk_label = record.get('risk_label', risk_level)
    recommendation = record.get('recommendation',
                                 'Please consult a qualified healthcare professional.')

    result_data = [
        [Paragraph(f'<b>Risk Level</b>', body_style),
         Paragraph(f'<font color="{text_c}"><b>{risk_label}</b></font>', body_style)],
        [Paragraph('<b>Probability Score</b>', body_style),
         Paragraph(f'<b>{prob_pct:.1f}%</b>', body_style)],
        [Paragraph('<b>Recommendation</b>', body_style),
         Paragraph(recommendation, body_style)],
    ]
    result_tbl = Table(result_data, colWidths=[3.5 * cm, (W - 3 * cm - 3.5 * cm)])
    result_tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, -1), colors.HexColor(bg)),
        ('LINEABOVE',   (0, 0), (-1, 0),  1.5, colors.HexColor(border)),
        ('LINEBELOW',   (0, -1), (-1, -1), 1.5, colors.HexColor(border)),
        ('LINEBEFORE',  (0, 0), (0, -1),  1.5, colors.HexColor(border)),
        ('LINEAFTER',   (-1, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING',     (0, 0), (-1, -1), 6),
        ('VALIGN',      (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(result_tbl)
    story.append(Spacer(1, 10))

    # ── Images (original + Grad-CAM) ─────────────────────────────────────────
    img_row = []
    img_size = 6.5 * cm

    if image_bytes:
        try:
            rl_img = RLImage(io.BytesIO(image_bytes), width=img_size, height=img_size)
            img_row.append([Paragraph('<b>Uploaded Image</b>', body_style), rl_img])
        except Exception:
            pass

    if gradcam_bytes:
        try:
            rl_gcam = RLImage(io.BytesIO(gradcam_bytes), width=img_size, height=img_size)
            img_row.append([Paragraph('<b>Grad-CAM Heatmap</b><br/>'
                                       '<font size="7" color="#64748b">'
                                       'Red = highest model attention</font>',
                                       body_style), rl_gcam])
        except Exception:
            pass

    if img_row:
        story.append(Paragraph("Visual Analysis", section_style))
        n = len(img_row)
        col_w = (W - 3 * cm) / n
        img_tbl = Table(
            [[r[0] for r in img_row], [r[1] for r in img_row]],
            colWidths=[col_w] * n
        )
        img_tbl.setStyle(TableStyle([
            ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',  (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('BOX',     (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('GRID',    (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ]))
        story.append(img_tbl)
        story.append(Spacer(1, 8))

    # ── Risk factors echoed back ──────────────────────────────────────────────
    meta_used = record.get('metadata_used', {})
    if meta_used:
        story.append(Paragraph("Reported Risk Factors", section_style))
        meta_rows = [[Paragraph(f'<b>{k.replace("_", " ").title()}</b>', body_style),
                      Paragraph(str(v), body_style)]
                     for k, v in meta_used.items()]
        if meta_rows:
            meta_tbl = Table(meta_rows, colWidths=[5 * cm, (W - 3 * cm - 5 * cm)])
            meta_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f9ff')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1),
                 [colors.HexColor('#fafafa'), colors.white]),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]))
            story.append(meta_tbl)
            story.append(Spacer(1, 6))

    # ── Disclaimer footer ─────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5,
                              color=colors.HexColor('#e2e8f0'), spaceBefore=8))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report was generated by an AI-assisted screening tool "
        "for educational and preliminary screening purposes only. "
        "It does NOT constitute a medical diagnosis. "
        "All findings must be reviewed and confirmed by a qualified healthcare professional "
        "before any clinical decision is made.",
        ParagraphStyle('Disc', parent=styles['Normal'], fontSize=7.5,
                        textColor=colors.HexColor('#6b7280'), leading=11)
    ))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"CuraLens AI Screening Platform  |  For Research & Training Use Only",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7,
                        textColor=colors.HexColor('#9ca3af'), alignment=TA_CENTER,
                        spaceBefore=4)
    ))

    doc.build(story)
    buf.seek(0)
    return buf


def preprocess_oral_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    
    return np.expand_dims(image, axis=0)

def get_threshold_for_mode(mode, optimal_threshold=0.512):
    if mode == 'screening':
        return optimal_threshold * 0.7
    else:
        return optimal_threshold

def get_oral_recommendation(cancer_prob, is_cancer, mode):
    if is_cancer:
        if cancer_prob > 0.7:
            return "HIGH RISK PATTERN: Significant abnormal tissue patterns detected. Immediate clinical evaluation and specialist consultation recommended."
        elif cancer_prob > 0.5:
            return "MODERATE RISK PATTERN: Abnormal tissue patterns observed. Specialist evaluation and diagnostic follow-up recommended."
        else:
            return "LOW RISK PATTERN: Mild abnormal patterns noted. Clinical examination and follow-up in 2-3 months recommended."
    else:
        if cancer_prob < 0.2:
            return "NORMAL PATTERN: Tissue appears within normal limits. Routine annual screening recommended."
        elif cancer_prob < 0.4:
            return "BENIGN PATTERN: Low probability of abnormality. Follow-up in 6-12 months recommended."
        else:
            return "BORDERLINE PATTERN: Requires monitoring. Follow-up in 3-6 months recommended."

# ---------------------------------------------------------------------------
# File-type validation helper
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'
}

def _allowed_image(filename: str) -> bool:
    """Return True if the filename has an allowed image extension."""
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_IMAGE_EXTENSIONS

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CuraLens – AI-Assisted Medical Screening Platform</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #60a5fa;
            --secondary: #7c3aed;
            --secondary-light: #a78bfa;
            --success: #10b981;
            --success-light: #34d399;
            --warning: #f59e0b;
            --warning-light: #fbbf24;
            --danger: #ef4444;
            --danger-light: #f87171;
            --dark: #0f172a;
            --dark-light: #1e293b;
            --light: #f8fafc;
            --gray-50: #f8fafc;
            --gray-100: #f1f5f9;
            --gray-200: #e2e8f0;
            --gray-300: #cbd5e1;
            --gray-400: #94a3b8;
            --gray-500: #64748b;
            --gray-600: #475569;
            --gray-700: #334155;
            --gray-800: #1e293b;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            --shadow-glow: 0 0 40px rgba(37, 99, 235, 0.15);
            --radius-sm: 8px;
            --radius: 12px;
            --radius-lg: 16px;
            --radius-xl: 24px;
            --radius-full: 9999px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-fast: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--dark);
            min-height: 100vh;
            line-height: 1.6;
            overflow-x: hidden;
        }

        /* Animated Background */
        .bg-animated {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f5f3ff 100%);
        }

        .bg-animated::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 20% 80%, rgba(37, 99, 235, 0.08) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
                        radial-gradient(circle at 40% 40%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
            animation: bgFloat 20s ease-in-out infinite;
        }

        @keyframes bgFloat {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(2%, 2%) rotate(1deg); }
            50% { transform: translate(-1%, 3%) rotate(-1deg); }
            75% { transform: translate(3%, -2%) rotate(0.5deg); }
        }

        /* Floating Orbs */
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(60px);
            opacity: 0.4;
            animation: orbFloat 15s ease-in-out infinite;
            z-index: -1;
        }

        .orb-1 {
            width: 400px;
            height: 400px;
            background: linear-gradient(135deg, var(--primary-light), var(--secondary-light));
            top: -100px;
            right: -100px;
            animation-delay: 0s;
        }

        .orb-2 {
            width: 300px;
            height: 300px;
            background: linear-gradient(135deg, var(--success-light), var(--primary-light));
            bottom: -50px;
            left: -50px;
            animation-delay: -5s;
        }

        .orb-3 {
            width: 250px;
            height: 250px;
            background: linear-gradient(135deg, var(--warning-light), var(--danger-light));
            top: 50%;
            left: 50%;
            animation-delay: -10s;
        }

        @keyframes orbFloat {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(30px, -30px) scale(1.1); }
            66% { transform: translate(-20px, 20px) scale(0.9); }
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 24px;
            position: relative;
            z-index: 1;
        }

        header {
            text-align: center;
            margin-bottom: 50px;
            animation: fadeInDown 0.8s ease-out;
        }

        .logo {
            display: inline-flex;
            align-items: center;
            gap: 16px;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 20px 40px;
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-lg), var(--shadow-glow);
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            transition: var(--transition);
        }

        .logo:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl), 0 0 60px rgba(37, 99, 235, 0.2);
        }

        .logo-icon {
            color: var(--primary);
            font-size: 32px;
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        h1 {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--primary-light) 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
            letter-spacing: -0.02em;
            animation: gradientShift 5s ease infinite;
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% center; }
            50% { background-position: 100% center; }
        }

        .subtitle {
            color: var(--gray-600);
            font-size: 1.15rem;
            font-weight: 400;
            max-width: 650px;
            margin: 0 auto;
            line-height: 1.7;
        }

        .modules-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 32px;
            margin-bottom: 50px;
            align-items: start;   /* FIX: each card only as tall as its own content */
        }

        @media (max-width: 1024px) {
            .modules-container {
                grid-template-columns: 1fr;
                gap: 24px;
            }
        }

        .module-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: var(--radius-xl);
            padding: 32px;
            box-shadow: var(--shadow-lg);
            transition: var(--transition);
            border: 1px solid rgba(255, 255, 255, 0.8);
            display: flex;
            flex-direction: column;
            height: 100%;
            position: relative;
            overflow: hidden;
        }

        .module-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            border-radius: var(--radius-xl) var(--radius-xl) 0 0;
        }

        .module-card.oral::before {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }

        .module-card.skin::before {
            background: linear-gradient(90deg, var(--warning), var(--danger));
        }

        .module-card:hover {
            box-shadow: var(--shadow-xl);
            transform: translateY(-8px);
        }

        .module-header {
            display: flex;
            align-items: center;
            gap: 18px;
            margin-bottom: 28px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--gray-200);
        }

        .module-icon {
            width: 64px;
            height: 64px;
            border-radius: var(--radius-lg);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.9rem;
            color: white;
            box-shadow: var(--shadow-md);
            transition: var(--transition);
        }

        .module-icon.oral {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
        }

        .module-icon.skin {
            background: linear-gradient(135deg, var(--warning), var(--danger));
        }

        .module-card:hover .module-icon {
            transform: scale(1.05) rotate(5deg);
        }

        .module-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--dark);
            letter-spacing: -0.01em;
        }

        .module-description {
            color: var(--gray-600);
            font-size: 0.95rem;
            margin-bottom: 28px;
            line-height: 1.65;
        }

        .upload-area {
            border: 2px dashed var(--gray-300);
            border-radius: var(--radius-lg);
            padding: 40px 24px;
            text-align: center;
            cursor: pointer;
            transition: var(--transition);
            background: rgba(248, 250, 252, 0.6);
            position: relative;
            overflow: hidden;
            margin-bottom: 28px;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .upload-area:hover {
            border-color: var(--primary);
            background: rgba(255, 255, 255, 0.9);
            box-shadow: var(--shadow-md);
        }

        .upload-area.dragover {
            border-color: var(--primary);
            background: rgba(37, 99, 235, 0.05);
            border-style: solid;
            transform: scale(1.02);
        }

        .upload-area.has-image {
            padding: 16px;
            min-height: auto;
        }

        .upload-preview {
            max-width: 100%;
            max-height: 200px;
            border-radius: var(--radius);
            box-shadow: var(--shadow-md);
            margin-bottom: 12px;
            display: none;
        }

        .upload-preview.visible {
            display: block;
        }

        .upload-icon {
            font-size: 56px;
            margin-bottom: 16px;
            opacity: 0.7;
            transition: var(--transition);
        }

        .upload-area:hover .upload-icon {
            transform: translateY(-4px);
            opacity: 1;
        }

        .upload-area.oral .upload-icon {
            color: var(--primary);
        }

        .upload-area.skin .upload-icon {
            color: var(--warning);
        }

        .upload-text h3 {
            color: var(--dark);
            margin-bottom: 8px;
            font-size: 1.25rem;
            font-weight: 600;
        }

        .upload-text p {
            color: var(--gray-500);
            font-size: 0.9rem;
            margin-bottom: 4px;
        }

        .file-input {
            display: none;
        }

        #oralFileName, #skinFileName {
            margin-top: 12px;
            font-weight: 500;
            font-size: 0.95rem;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: var(--radius-full);
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        #oralFileName {
            color: var(--primary);
        }

        #skinFileName {
            color: var(--warning);
        }

        .file-info {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 12px;
            padding: 10px 16px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: var(--radius-full);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .file-info i {
            font-size: 1rem;
        }

        .file-info.oral {
            color: var(--primary);
        }

        .file-info.skin {
            color: var(--warning);
        }

        .analyze-btn {
            background: none;
            border-radius: var(--radius-lg);
            padding: 18px 24px;
            font-size: 1.1rem;
            font-weight: 600;
            width: 100%;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-top: auto;
            border: none;
            position: relative;
            overflow: hidden;
            letter-spacing: 0.01em;
        }

        .analyze-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: 0.5s;
        }

        .analyze-btn:hover:not(:disabled)::before {
            left: 100%;
        }

        .analyze-btn.oral {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            box-shadow: var(--shadow-md);
        }

        .analyze-btn.oral:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(37, 99, 235, 0.35);
        }

        .analyze-btn.skin {
            background: linear-gradient(135deg, var(--warning), var(--danger));
            color: white;
            box-shadow: var(--shadow-md);
        }

        .analyze-btn.skin:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(245, 158, 11, 0.35);
        }

        .analyze-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .analyze-btn i {
            font-size: 1.2rem;
            transition: var(--transition);
        }

        .analyze-btn:hover:not(:disabled) i {
            transform: scale(1.1);
        }

        .loading-container {
            text-align: center;
            padding: 50px 24px;
            display: none;
            margin-top: 24px;
            background: rgba(255, 255, 255, 0.6);
            border-radius: var(--radius-lg);
            backdrop-filter: blur(10px);
        }

        .spinner {
            width: 56px;
            height: 56px;
            margin: 0 auto 24px;
            position: relative;
        }

        .spinner::before,
        .spinner::after {
            content: '';
            position: absolute;
            border-radius: 50%;
        }

        .spinner::before {
            width: 100%;
            height: 100%;
            border: 4px solid var(--gray-200);
        }

        .spinner::after {
            width: 100%;
            height: 100%;
            border: 4px solid transparent;
            border-top-color: var(--primary);
            animation: spin 1s cubic-bezier(0.5, 0, 0.5, 1) infinite;
        }

        .oral .spinner::after {
            border-top-color: var(--primary);
        }

        .skin .spinner::after {
            border-top-color: var(--warning);
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 8px;
        }

        .loading-subtext {
            font-size: 0.9rem;
            color: var(--gray-500);
        }

        .loading-dots {
            display: inline-flex;
            gap: 4px;
            margin-left: 4px;
        }

        .loading-dots span {
            width: 6px;
            height: 6px;
            background: var(--primary);
            border-radius: 50%;
            animation: bounce 1.4s ease-in-out infinite both;
        }

        .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
        .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
        .loading-dots span:nth-child(3) { animation-delay: 0s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .results-container {
            display: none;
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
            margin-top: 28px;
        }

        @keyframes slideUp {
            from { 
                opacity: 0; 
                transform: translateY(30px); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }

        .result-header {
            color: white;
            padding: 28px 32px;
            border-radius: var(--radius-xl) var(--radius-xl) 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
            overflow: hidden;
        }

        .result-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%);
        }

        .oral .result-header {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
        }

        .skin .result-header {
            background: linear-gradient(135deg, var(--warning), var(--danger));
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
            position: relative;
            z-index: 1;
        }

        .result-title i {
            font-size: 1.3rem;
        }

        .risk-badge {
            padding: 10px 24px;
            border-radius: var(--radius-full);
            font-weight: 700;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            background: white;
            position: relative;
            z-index: 1;
            box-shadow: var(--shadow-md);
            transition: var(--transition);
        }

        .risk-badge:hover {
            transform: scale(1.05);
        }

        .oral .risk-badge.very-low, .skin .risk-badge.very-low { color: var(--success); }
        .oral .risk-badge.low, .skin .risk-badge.low { color: var(--success); }
        .oral .risk-badge.moderate, .skin .risk-badge.moderate { color: var(--warning); }
        .oral .risk-badge.high, .skin .risk-badge.high { color: var(--danger); }
        .oral .risk-badge.borderline, .skin .risk-badge.borderline { color: var(--gray-600); }

        .result-content {
            padding: 32px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 0 0 var(--radius-xl) var(--radius-xl);
            border: 2px solid var(--gray-100);
            border-top: none;
            backdrop-filter: blur(10px);
        }

        .probability-section {
            margin-bottom: 32px;
        }

        .probability-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
            font-weight: 600;
            color: var(--dark);
            font-size: 1rem;
        }

        .probability-value {
            font-size: 1.1rem;
            font-weight: 700;
        }

        .oral .probability-value { color: var(--primary); }
        .skin .probability-value { color: var(--warning); }

        .probability-bar {
            height: 16px;
            background: var(--gray-100);
            border-radius: var(--radius-full);
            overflow: hidden;
            margin-bottom: 8px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }

        .probability-fill {
            height: 100%;
            border-radius: var(--radius-full);
            width: 0%;
            transition: width 1.2s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
        }

        .probability-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .oral .probability-fill {
            background: linear-gradient(90deg, var(--success) 0%, var(--warning) 50%, var(--danger) 100%);
        }

        .skin .probability-fill {
            background: linear-gradient(90deg, var(--success) 0%, var(--warning) 50%, var(--danger) 100%);
        }

        .probability-scale {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--gray-500);
            font-weight: 500;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin: 28px 0;
        }

        .metric-box {
            background: linear-gradient(135deg, var(--gray-50), var(--gray-100));
            border-radius: var(--radius-lg);
            padding: 24px;
            text-align: center;
            transition: var(--transition);
            border: 1px solid var(--gray-100);
        }

        .metric-box:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-md);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 6px;
            letter-spacing: -0.02em;
        }

        .oral .metric-value {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .skin .metric-value {
            background: linear-gradient(135deg, var(--warning), var(--danger));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .metric-label {
            font-size: 0.8rem;
            color: var(--gray-500);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }

        .recommendation-box {
            border-radius: var(--radius-lg);
            padding: 28px;
            margin-top: 28px;
            border-left: 4px solid;
            position: relative;
            overflow: hidden;
        }

        .recommendation-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            opacity: 0.5;
        }

        .oral .recommendation-box {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(124, 58, 237, 0.08));
            border-left-color: var(--primary);
        }

        .skin .recommendation-box {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(239, 68, 68, 0.08));
            border-left-color: var(--warning);
        }

        .recommendation-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
            font-weight: 700;
            color: var(--dark);
            font-size: 1.1rem;
            position: relative;
            z-index: 1;
        }

        .recommendation-header i {
            font-size: 1.2rem;
        }

        .oral .recommendation-header i { color: var(--primary); }
        .skin .recommendation-header i { color: var(--warning); }

        .recommendation-box p {
            color: var(--gray-700);
            font-size: 0.95rem;
            line-height: 1.7;
            position: relative;
            z-index: 1;
        }

        .disclaimer {
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-top: 28px;
            border-left: 4px solid var(--danger);
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.06), rgba(245, 158, 11, 0.06));
            position: relative;
        }

        .disclaimer-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
            color: var(--danger);
            font-weight: 700;
            font-size: 1rem;
        }

        .disclaimer p {
            color: var(--gray-600);
            font-size: 0.9rem;
            line-height: 1.65;
        }

        .platform-stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-top: 50px;
        }

        @media (max-width: 1024px) {
            .platform-stats {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 640px) {
            .platform-stats {
                grid-template-columns: 1fr;
            }
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 28px 24px;
            border-radius: var(--radius-xl);
            text-align: center;
            box-shadow: var(--shadow-lg);
            border: 1px solid rgba(255, 255, 255, 0.8);
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary), var(--secondary), var(--success));
        }

        .stat-card:hover {
            transform: translateY(-6px);
            box-shadow: var(--shadow-xl);
        }

        .stat-icon {
            width: 48px;
            height: 48px;
            margin: 0 auto 16px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: var(--radius-lg);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.3rem;
        }

        .stat-value {
            font-size: 2.25rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }

        .stat-label {
            color: var(--gray-500);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        footer {
            text-align: center;
            margin-top: 60px;
            padding: 32px 24px;
            border-top: 1px solid var(--gray-200);
            color: var(--gray-500);
            font-size: 0.9rem;
        }

        footer p {
            margin-bottom: 8px;
        }

        footer p:last-child {
            margin-bottom: 0;
        }

        .footer-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: var(--radius-full);
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--gray-600);
            margin-top: 12px;
        }

        .footer-badge i {
            color: var(--primary);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .placeholder {
            text-align: center;
            padding: 50px 24px;
            color: var(--gray-500);
            border: 2px dashed var(--gray-200);
            border-radius: var(--radius-lg);
            margin-top: 24px;
            background: rgba(248, 250, 252, 0.5);
        }

        .placeholder i {
            font-size: 56px;
            margin-bottom: 16px;
            opacity: 0.4;
            color: var(--gray-400);
        }

        .placeholder h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--gray-600);
            margin-bottom: 8px;
        }

        .placeholder p {
            font-size: 0.95rem;
            color: var(--gray-500);
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: var(--gray-100);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--gray-300);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--gray-400);
        }

        /* Selection Styling */
        ::selection {
            background: rgba(37, 99, 235, 0.2);
            color: var(--dark);
        }

        /* Focus Styling */
        *:focus {
            outline: 2px solid var(--primary);
            outline-offset: 2px;
        }

        button:focus {
            outline-offset: 4px;
        }

        /* ── Version toggle ─────────────────────────────── */
        .version-toggle {
            display: flex;
            align-items: center;
            background: var(--gray-100);
            border-radius: var(--radius-full);
            padding: 4px;
            width: fit-content;
            margin-bottom: 20px;
            border: 1px solid var(--gray-200);
        }
        .v-btn {
            padding: 8px 22px;
            border-radius: var(--radius-full);
            font-size: 0.87rem;
            font-weight: 600;
            cursor: pointer;
            border: none;
            background: transparent;
            color: var(--gray-500);
            transition: var(--transition-fast);
            letter-spacing: 0.3px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .v-btn.v1-active {
            background: white;
            color: var(--primary);
            box-shadow: var(--shadow-sm);
        }
        .v-btn.v2-active {
            background: white;
            color: var(--secondary);
            box-shadow: var(--shadow-sm);
        }
        .v-btn .v-pip {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: currentColor;
            opacity: 0.5;
        }
        .v-btn.v1-active .v-pip, .v-btn.v2-active .v-pip { opacity: 1; }

        /* ── Metadata panel ─────────────────────────────── */
        .metadata-panel {
            background: rgba(124, 58, 237, 0.04);
            border: 1px solid rgba(124, 58, 237, 0.18);
            border-radius: var(--radius-lg);
            padding: 18px 20px;
            margin-bottom: 20px;
            display: none;
            animation: fadeIn 0.3s ease;
        }
        .metadata-panel.visible { display: block; }
        .metadata-panel-title {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.3px;
            color: var(--secondary);
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        .meta-field label {
            display: block;
            font-size: 0.79rem;
            font-weight: 600;
            color: var(--gray-600);
            margin-bottom: 5px;
        }
        .meta-field input,
        .meta-field select {
            width: 100%;
            padding: 8px 11px;
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-sm);
            font-size: 0.9rem;
            font-family: inherit;
            background: white;
            color: var(--dark);
            transition: var(--transition-fast);
            -webkit-appearance: none;
        }
        .meta-field input:focus,
        .meta-field select:focus {
            border-color: var(--secondary);
            outline: none;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
        }

        /* ── Error banner ────────────────────────────────── */
        .error-banner {
            display: none;
            align-items: flex-start;
            gap: 12px;
            padding: 14px 16px;
            background: rgba(239, 68, 68, 0.05);
            border: 1px solid rgba(239, 68, 68, 0.25);
            border-left: 4px solid var(--danger);
            border-radius: var(--radius-lg);
            margin-top: 14px;
            animation: fadeIn 0.3s ease;
        }
        .error-banner.visible { display: flex; }
        .error-banner > i { color: var(--danger); font-size: 1rem; margin-top: 2px; flex-shrink: 0; }
        .error-banner-body { flex: 1; }
        .error-banner-title { font-weight: 700; color: var(--danger); font-size: 0.85rem; margin-bottom: 2px; }
        .error-banner-msg  { color: var(--gray-600); font-size: 0.85rem; line-height: 1.45; }
        .error-banner-close {
            background: none; border: none; cursor: pointer;
            color: var(--gray-400); font-size: 1rem; padding: 0 2px;
            margin-left: auto; flex-shrink: 0;
        }
        .error-banner-close:hover { color: var(--danger); }

        /* ── Risk badge — filled variant ─────────────────── */
        .risk-badge.filled { color: white !important; font-size: 0.83rem; }
        .risk-badge.filled.low,
        .risk-badge.filled.very-low  { background: var(--success) !important; }
        .risk-badge.filled.medium,
        .risk-badge.filled.moderate  { background: var(--warning) !important; }
        .risk-badge.filled.high      { background: var(--danger)  !important; }
        .risk-badge.filled.borderline{ background: var(--gray-500)!important; }

        /* ── GradCAM panel ───────────────────────────────── */
        .gradcam-panel {
            margin-top: 24px;
            padding-top: 20px;
            border-top: 1px solid var(--gray-200);
        }
        .gradcam-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.79rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.3px;
            color: var(--secondary);
            margin-bottom: 12px;
        }
        .gradcam-images {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .gradcam-image-box {
            border-radius: var(--radius-lg);
            overflow: hidden;
            background: var(--gray-100);
            border: 1px solid var(--gray-200);
            transition: var(--transition-fast);
        }
        .gradcam-image-box:hover { box-shadow: var(--shadow-md); }
        .gradcam-image-box img {
            width: 100%; height: 180px;
            object-fit: cover; display: block;
        }
        .gradcam-image-label {
            padding: 7px 12px;
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--gray-600);
            text-align: center;
            background: white;
            border-top: 1px solid var(--gray-100);
        }

        /* ── V2 tag badge ────────────────────────────────── */
        .v2-tag {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            color: white;
            border-radius: var(--radius-full);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            vertical-align: middle;
            margin-left: 7px;
        }

        /* ── Metadata result chips ───────────────────────── */
        .meta-result-row {
            display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0 0;
        }
        .meta-result-chip {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 11px;
            background: rgba(124, 58, 237, 0.07);
            border: 1px solid rgba(124, 58, 237, 0.18);
            border-radius: var(--radius-full);
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--secondary);
        }

        /* ── Tooltip icon ────────────────────────────────── */
        .tooltip-icon {
            display: inline-block;
            width: 15px; height: 15px;
            line-height: 15px;
            text-align: center;
            background: var(--gray-200);
            color: var(--gray-600);
            border-radius: 50%;
            font-size: 0.72rem;
            cursor: help;
            position: relative;
        }
        .tooltip-icon:hover::after {
            content: attr(title);
            position: absolute;
            left: 50%; bottom: calc(100% + 6px);
            transform: translateX(-50%);
            background: var(--dark);
            color: white;
            font-size: 0.72rem;
            padding: 6px 10px;
            border-radius: var(--radius-sm);
            white-space: normal;
            min-width: 180px;
            max-width: 260px;
            z-index: 9999;
            pointer-events: none;
            box-shadow: var(--shadow-md);
        }

        /* ── Clinical interpretation ─────────────────────── */
        .clinical-interpretation {
            background: linear-gradient(135deg,
                rgba(37, 99, 235, 0.04) 0%, rgba(124, 58, 237, 0.04) 100%);
            border: 1px solid rgba(37, 99, 235, 0.15);
            border-left: 4px solid var(--primary);
            border-radius: var(--radius-lg);
            padding: 14px 16px;
            margin-top: 16px;
        }
        .clinical-interp-header {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--primary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .clinical-interpretation p {
            font-size: 0.88rem;
            color: var(--gray-700);
            line-height: 1.65;
            margin: 0;
        }

        /* ── Grad-CAM opacity slider row ─────────────────── */
        .gradcam-opacity-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }

        /* ── Metadata warnings ───────────────────────────── */
        .meta-warnings {
            background: rgba(245, 158, 11, 0.06);
            border: 1px solid rgba(245, 158, 11, 0.25);
            border-radius: var(--radius);
            padding: 10px 14px;
            margin-top: 12px;
        }

        /* ═══════════════════════════════════════════════════
           FIELD SCREENING MODULE  (green theme)
           ═══════════════════════════════════════════════════ */
        .module-card.field::before {
            background: linear-gradient(90deg, #16a34a, #059669);
        }

        .module-icon.field {
            background: linear-gradient(135deg, #16a34a, #059669);
        }

        .upload-area.field .upload-icon { color: #16a34a; }
        .upload-area.field:hover        { border-color: #16a34a; background: rgba(22,163,74,0.04); }
        .upload-area.field.dragover     { border-color: #16a34a; background: rgba(22,163,74,0.08); }

        .analyze-btn.field {
            background: linear-gradient(135deg, #16a34a, #059669);
            color: white;
            box-shadow: 0 4px 14px rgba(22,163,74,0.30);
        }
        .analyze-btn.field:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(22,163,74,0.40);
        }

        /* Patient info form grid */
        .field-form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px 14px;
            margin-bottom: 16px;
        }
        @media (max-width: 600px) { .field-form-grid { grid-template-columns: 1fr; } }

        .field-form-grid .meta-field label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--gray-600);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
            display: block;
        }
        .field-form-grid .meta-field input,
        .field-form-grid .meta-field select {
            width: 100%;
            padding: 9px 12px;
            border: 1.5px solid rgba(22,163,74,0.25);
            border-radius: var(--radius);
            background: rgba(255,255,255,0.7);
            font-size: 0.92rem;
            color: var(--gray-800);
            outline: none;
            transition: border-color 0.2s;
            box-sizing: border-box;
        }
        .field-form-grid .meta-field input:focus,
        .field-form-grid .meta-field select:focus {
            border-color: #16a34a;
        }

        /* Big risk result card */
        .field-result-card {
            border-radius: var(--radius-lg);
            padding: 24px 20px 20px;
            text-align: center;
            margin-bottom: 16px;
            transition: var(--transition);
        }
        .field-result-card.risk-high   { background: #fef2f2; border: 2px solid #ef4444; }
        .field-result-card.risk-medium { background: #fffbeb; border: 2px solid #f59e0b; }
        .field-result-card.risk-low    { background: #f0fdf4; border: 2px solid #16a34a; }

        .field-risk-icon { font-size: 3rem; margin-bottom: 8px; }
        .field-risk-label {
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .risk-high   .field-risk-label { color: #dc2626; }
        .risk-medium .field-risk-label { color: #d97706; }
        .risk-low    .field-risk-label { color: #16a34a; }

        .field-risk-action {
            font-size: 1.05rem;
            font-weight: 700;
            padding: 8px 16px;
            border-radius: 999px;
            display: inline-block;
            margin: 6px 0 8px;
        }
        .risk-high   .field-risk-action { background:#fecaca; color:#991b1b; }
        .risk-medium .field-risk-action { background:#fde68a; color:#92400e; }
        .risk-low    .field-risk-action { background:#bbf7d0; color:#14532d; }

        .field-prob-text {
            font-size: 0.88rem;
            color: var(--gray-600);
            margin-top: 4px;
        }

        /* Download report button */
        .report-download-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            width: 100%;
            padding: 14px 20px;
            background: linear-gradient(135deg, #1e40af, #3b82f6);
            color: white;
            border: none;
            border-radius: var(--radius-lg);
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            transition: var(--transition);
            margin-bottom: 12px;
        }
        .report-download-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(59,130,246,0.4); }

        /* Patient history mini-table */
        .history-toggle-btn {
            background: none;
            border: 1.5px solid rgba(22,163,74,0.35);
            border-radius: var(--radius);
            padding: 8px 14px;
            font-size: 0.85rem;
            color: #16a34a;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 8px;
            transition: var(--transition);
        }
        .history-toggle-btn:hover { background: rgba(22,163,74,0.06); }

        .history-table-wrap {
            display: none;
            margin-top: 12px;
            overflow-x: auto;
        }
        .history-table-wrap table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }
        .history-table-wrap th {
            background: rgba(22,163,74,0.08);
            color: var(--gray-700);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 7px 10px;
            text-align: left;
        }
        .history-table-wrap td {
            padding: 7px 10px;
            border-bottom: 1px solid rgba(0,0,0,0.06);
            color: var(--gray-700);
        }
        .history-table-wrap tr:hover td { background: rgba(22,163,74,0.04); }
        .hist-risk-high   { color:#dc2626; font-weight:700; }
        .hist-risk-medium { color:#d97706; font-weight:700; }
        .hist-risk-low    { color:#16a34a; font-weight:700; }
    </style>
</head>
<body>
    <!-- Animated Background -->
    <div class="bg-animated"></div>
    
    <!-- Floating Orbs -->
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    
    <div class="container">
        <header>
            <div class="logo">
                <i class="fas fa-microscope logo-icon"></i>
                <div>
                    <h1>CuraLens</h1>
                    <div style="font-size: 0.9rem; color: var(--primary); font-weight: 600;">AI-Assisted Multi-Module Screening Platform</div>
                </div>
            </div>
            <p class="subtitle">Advanced pattern recognition for early detection. Two specialized modules for comprehensive screening.</p>
        </header>

        <div class="modules-container">
            <!-- ORAL SCREENING MODULE -->
            <div class="module-card oral">
                <div class="module-header">
                    <div class="module-icon oral">
                        <i class="fas fa-tooth"></i>
                    </div>
                    <div>
                        <div class="module-title">🦷 Oral Screening Module <span class="v2-tag">+V2</span></div>
                        <div style="color: var(--gray-700); font-size: 0.95rem;">AI-Powered Oral Tissue Analysis</div>
                    </div>
                </div>

                <div class="module-description">
                    Upload intraoral images for automated screening of abnormal tissue patterns. Supports JPG, PNG formats. Images should be clear, well-lit, and focused on the oral mucosa.
                </div>

                <!-- Version toggle -->
                <div class="version-toggle" id="oralVersionToggle">
                    <button class="v-btn v1-active" id="vBtnV1" onclick="setVersion('v1')">
                        <span class="v-pip"></span> v1 &nbsp;Image-only
                    </button>
                    <button class="v-btn" id="vBtnV2" onclick="setVersion('v2')">
                        <span class="v-pip"></span> v2 &nbsp;Multimodal
                    </button>
                </div>

                <!-- v2 metadata inputs (shown when v2 is selected) -->
                <div class="metadata-panel" id="metadataPanel">
                    <div class="metadata-panel-title"><i class="fas fa-user-md"></i> Patient Risk Factors</div>

                    <!-- Cancer Type Selector -->
                    <div class="meta-field" style="grid-column: 1/-1; margin-bottom: 8px;">
                        <label for="metaCancerType" style="display:flex;align-items:center;gap:6px;">
                            Cancer Type
                            <span class="tooltip-icon" title="Select the cancer type to determine which AI model and metadata fields to use.">&#9432;</span>
                        </label>
                        <select id="metaCancerType" onchange="onCancerTypeChange(this.value)">
                            <option value="oral_legacy">Oral Cancer (Classic — 4 fields)</option>
                            <option value="oral">Oral Cancer (Clinical — 6 fields)</option>
                            <option value="skin">Skin Cancer (Clinical — 6 fields)</option>
                        </select>
                    </div>

                    <!-- ORAL LEGACY fields (default visible) -->
                    <div class="meta-grid" id="metaGroupOralLegacy">
                        <div class="meta-field">
                            <label for="metaAge">Age (years)</label>
                            <input type="number" id="metaAge" min="1" max="120" placeholder="e.g. 45" value="45">
                        </div>
                        <div class="meta-field">
                            <label for="metaSunExposure" style="display:flex;align-items:center;gap:6px;">
                                Sun Exposure (0–10)
                                <span class="tooltip-icon" title="0 = minimal, 10 = extreme daily sun exposure">&#9432;</span>
                            </label>
                            <input type="number" id="metaSunExposure" min="0" max="10" step="0.5" placeholder="0–10" value="3">
                        </div>
                        <div class="meta-field">
                            <label for="metaSmoking">Smoking</label>
                            <select id="metaSmoking">
                                <option value="0">Non-smoker</option>
                                <option value="1">Smoker</option>
                            </select>
                        </div>
                        <div class="meta-field">
                            <label for="metaAlcohol">Alcohol Use</label>
                            <select id="metaAlcohol">
                                <option value="0">Non-drinker</option>
                                <option value="1">Regular drinker</option>
                            </select>
                        </div>
                    </div>

                    <!-- ORAL CLINICAL fields (hidden by default) -->
                    <div class="meta-grid" id="metaGroupOral" style="display:none;">
                        <div class="meta-field">
                            <label for="oralAge">Age (years)</label>
                            <input type="number" id="oralAge" min="18" max="120" placeholder="e.g. 52" value="52">
                        </div>
                        <div class="meta-field">
                            <label for="oralSmokingYears" style="display:flex;align-items:center;gap:6px;">
                                Smoking Years
                                <span class="tooltip-icon" title="Total years as a smoker. Enter 0 if never smoked.">&#9432;</span>
                            </label>
                            <input type="number" id="oralSmokingYears" min="0" max="80" placeholder="0 if never" value="0">
                        </div>
                        <div class="meta-field">
                            <label for="oralCigsPerDay" style="display:flex;align-items:center;gap:6px;">
                                Cigarettes/Day
                                <span class="tooltip-icon" title="Average cigarettes smoked per day. Enter 0 if non-smoker.">&#9432;</span>
                            </label>
                            <input type="number" id="oralCigsPerDay" min="0" max="100" placeholder="0 if non-smoker" value="0">
                        </div>
                        <div class="meta-field">
                            <label for="oralAlcoholUnits" style="display:flex;align-items:center;gap:6px;">
                                Alcohol Units/Week
                                <span class="tooltip-icon" title="1 unit = 10ml pure alcohol (e.g. half pint of beer ≈ 1 unit, glass of wine ≈ 2 units)">&#9432;</span>
                            </label>
                            <input type="number" id="oralAlcoholUnits" min="0" max="200" placeholder="e.g. 14" value="0">
                        </div>
                        <div class="meta-field">
                            <label for="oralChewingTobacco">Chewing Tobacco / Betel Nut</label>
                            <select id="oralChewingTobacco">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        <div class="meta-field">
                            <label for="oralFamilyHistory">Family History of Oral Cancer</label>
                            <select id="oralFamilyHistory">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                    </div>

                    <!-- SKIN CLINICAL fields (hidden by default) -->
                    <div class="meta-grid" id="metaGroupSkin" style="display:none;">
                        <div class="meta-field">
                            <label for="skinAge">Age (years)</label>
                            <input type="number" id="skinAge" min="18" max="120" placeholder="e.g. 45" value="45">
                        </div>
                        <div class="meta-field">
                            <label for="skinType" style="display:flex;align-items:center;gap:6px;">
                                Fitzpatrick Skin Type
                                <span class="tooltip-icon" title="1=Very Fair (always burns), 2=Fair (usually burns), 3=Medium (sometimes burns), 4=Olive (rarely burns), 5=Brown (very rarely burns), 6=Dark (never burns)">&#9432;</span>
                            </label>
                            <select id="skinType">
                                <option value="1">Type I – Very Fair</option>
                                <option value="2">Type II – Fair</option>
                                <option value="3" selected>Type III – Medium</option>
                                <option value="4">Type IV – Olive</option>
                                <option value="5">Type V – Brown</option>
                                <option value="6">Type VI – Dark</option>
                            </select>
                        </div>
                        <div class="meta-field">
                            <label for="skinSunburns" style="display:flex;align-items:center;gap:6px;">
                                Significant Sunburns (lifetime)
                                <span class="tooltip-icon" title="Number of significant sunburns (blistering/peeling) in your lifetime.">&#9432;</span>
                            </label>
                            <input type="number" id="skinSunburns" min="0" max="50" placeholder="e.g. 3" value="2">
                        </div>
                        <div class="meta-field">
                            <label for="skinOutdoorHours" style="display:flex;align-items:center;gap:6px;">
                                Outdoor Hours/Week
                                <span class="tooltip-icon" title="Average hours spent outdoors per week, year-round.">&#9432;</span>
                            </label>
                            <input type="number" id="skinOutdoorHours" min="0" max="112" placeholder="e.g. 10" value="10">
                        </div>
                        <div class="meta-field">
                            <label for="skinTanningBed">Tanning Bed Use</label>
                            <select id="skinTanningBed">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        <div class="meta-field">
                            <label for="skinFamilyHistory">Family History of Skin Cancer</label>
                            <select id="skinFamilyHistory">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div class="upload-area oral" id="oralUploadArea">
                    <input type="file" id="oralFileInput" class="file-input" accept="image/*">
                    <img id="oralPreview" class="upload-preview" alt="Preview">
                    <div id="oralUploadContent">
                        <div class="upload-icon">
                            <i class="fas fa-cloud-upload-alt"></i>
                        </div>
                        <div class="upload-text">
                            <h3>Upload Oral Image</h3>
                            <p>JPG, PNG, or JPEG format (max 5MB)</p>
                            <p>Intraoral lesion images only</p>
                        </div>
                    </div>
                    <div id="oralFileName" class="file-info oral" style="display: none;">
                        <i class="fas fa-file-image"></i>
                        <span></span>
                    </div>
                </div>

                <button class="analyze-btn oral" onclick="analyzeOral()" id="oralAnalyzeBtn">
                    <i class="fas fa-play-circle"></i>
                    <span id="oralBtnText">Analyze Oral Image</span>
                </button>

                <!-- Error banner -->
                <div class="error-banner" id="oralErrorBanner">
                    <i class="fas fa-exclamation-circle"></i>
                    <div class="error-banner-body">
                        <div class="error-banner-title">Analysis Error</div>
                        <div class="error-banner-msg" id="oralErrorMsg">An error occurred.</div>
                    </div>
                    <button class="error-banner-close" onclick="hideError('oral')" title="Dismiss">&#10005;</button>
                </div>

                <div class="loading-container" id="oralLoading">
                    <div class="spinner"></div>
                    <p class="loading-text">Processing oral tissue patterns<span class="loading-dots"><span></span><span></span><span></span></span></p>
                    <p class="loading-subtext">Analyzing epithelial architecture and tissue morphology</p>
                </div>

                <div class="results-container oral" id="oralResults">
                    <div class="result-header">
                        <div class="result-title">
                            <i class="fas fa-tooth"></i>
                            Oral Screening Results
                        </div>
                        <div class="risk-badge" id="oralRiskBadge">--</div>
                    </div>
                    <div class="result-content">
                        <div class="probability-section">
                            <div class="probability-label">
                                <span>Abnormal Pattern Probability</span>
                                <span id="oralProbabilityValue">0%</span>
                            </div>
                            <div class="probability-bar">
                                <div class="probability-fill" id="oralProbabilityFill"></div>
                            </div>
                            <div class="probability-scale">
                                <span>Normal</span>
                                <span>Borderline</span>
                                <span>Suspicious</span>
                            </div>
                        </div>

                        <div class="metrics-grid">
                            <div class="metric-box">
                                <div class="metric-value" id="oralPatternProb">--%</div>
                                <div class="metric-label">Pattern Score</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value" id="oralPrediction">--</div>
                                <div class="metric-label">Screening Result</div>
                            </div>
                        </div>

                        <div class="recommendation-box">
                            <div class="recommendation-header">
                                <i class="fas fa-clipboard-check"></i>
                                Clinical Guidance
                            </div>
                            <p id="oralRecommendation">Upload an oral image to receive screening guidance based on tissue pattern analysis.</p>
                        </div>

                        <div class="disclaimer">
                            <div class="disclaimer-header">
                                <i class="fas fa-exclamation-triangle"></i>
                                Important Notice
                            </div>
                            <p>This is an AI-assisted screening tool, NOT a diagnostic system. All results require confirmation by qualified healthcare professionals. Intended for screening purposes only.</p>
                        </div>
                    </div>
                </div>

                <!-- v2 results container (with GradCAM) -->
                <div class="results-container oral" id="oralV2Results">
                    <div class="result-header">
                        <div class="result-title">
                            <i class="fas fa-brain"></i>
                            Multimodal Results <span class="v2-tag" style="font-size:0.7rem;">v2</span>
                        </div>
                        <div class="risk-badge filled" id="oralV2RiskBadge">--</div>
                    </div>
                    <div class="result-content">
                        <div class="probability-section">
                            <div class="probability-label">
                                <span>Cancer Probability</span>
                                <span id="oralV2ProbValue">0%</span>
                            </div>
                            <div class="probability-bar">
                                <div class="probability-fill" id="oralV2ProbFill"></div>
                            </div>
                            <div class="probability-scale">
                                <span>Low Risk</span>
                                <span>Medium Risk</span>
                                <span>High Risk</span>
                            </div>
                        </div>

                        <!-- Metadata chips -->
                        <div class="meta-result-row" id="oralV2MetaChips"></div>

                        <div class="metrics-grid">
                            <div class="metric-box">
                                <div class="metric-value" id="oralV2PatternProb">--%</div>
                                <div class="metric-label">Pattern Score</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value" id="oralV2RiskLabel" style="font-size:1.4rem;">--</div>
                                <div class="metric-label">Risk Tier</div>
                            </div>
                        </div>

                        <!-- GradCAM side-by-side panel -->
                        <div class="gradcam-panel" id="oralGradcamPanel" style="display:none;">
                            <div class="gradcam-title">
                                <i class="fas fa-eye"></i> Grad-CAM Explainability
                                <span class="tooltip-icon" title="Gradient-weighted Class Activation Map highlights which image regions most influenced the model's prediction. Red/hot areas have the highest influence.">&#9432;</span>
                            </div>
                            <!-- Opacity slider -->
                            <div class="gradcam-opacity-row">
                                <label for="gradcamOpacity" style="font-size:0.82rem;color:var(--gray-600);">Heatmap Opacity</label>
                                <input type="range" id="gradcamOpacity" min="0" max="100" value="70"
                                       oninput="updateGradcamOpacity(this.value)"
                                       style="flex:1;accent-color:var(--primary);">
                                <span id="gradcamOpacityVal" style="font-size:0.82rem;color:var(--gray-600);min-width:32px;text-align:right;">70%</span>
                            </div>
                            <div class="gradcam-images">
                                <div class="gradcam-image-box">
                                    <img id="oralOriginalImg" alt="Original Image" style="display:none;">
                                    <div class="gradcam-image-label">Original Image</div>
                                </div>
                                <div class="gradcam-image-box" style="position:relative;">
                                    <img id="oralGradcamImg" alt="Grad-CAM Overlay" style="opacity:0.7;display:none;">
                                    <div class="gradcam-image-label">🔥 Activation Heatmap</div>
                                </div>
                            </div>
                        </div>

                        <div class="recommendation-box">
                            <div class="recommendation-header">
                                <i class="fas fa-clipboard-check"></i>
                                Clinical Guidance
                            </div>
                            <p id="oralV2Recommendation">Upload an oral image to receive screening guidance.</p>
                        </div>

                        <!-- Clinical interpretation paragraph (v3) -->
                        <div class="clinical-interpretation" id="oralV2ClinicalInterp" style="display:none;">
                            <div class="clinical-interp-header">
                                <i class="fas fa-stethoscope"></i> AI Clinical Interpretation
                            </div>
                            <p id="oralV2InterpText"></p>
                        </div>

                        <!-- Metadata warnings -->
                        <div class="meta-warnings" id="oralV2MetaWarnings" style="display:none;">
                            <div style="font-size:0.78rem;font-weight:600;color:var(--warning);margin-bottom:4px;">⚠️ Input Notes</div>
                            <ul id="oralV2WarningsList" style="font-size:0.78rem;color:var(--gray-600);padding-left:1.2em;margin:0;"></ul>
                        </div>

                        <div class="disclaimer">
                            <div class="disclaimer-header">
                                <i class="fas fa-exclamation-triangle"></i>
                                Important Notice
                            </div>
                            <p>Experimental v2 multimodal model — not validated for clinical use. Always consult a qualified healthcare professional.</p>
                        </div>
                    </div>
                </div>

                <div class="placeholder" id="oralPlaceholder">
                    <i class="fas fa-chart-line"></i>
                    <h3 style="margin-bottom: 10px;">Awaiting Oral Analysis</h3>
                    <p>Upload an intraoral image to begin tissue pattern screening</p>
                </div>
            </div>

            <!-- SKIN SCREENING MODULE -->
            <div class="module-card skin">
                <div class="module-header">
                    <div class="module-icon skin">
                        <i class="fas fa-hand"></i>
                    </div>
                    <div>
                        <div class="module-title">🧴 Skin Screening Module <span class="v2-tag">+V2</span></div>
                        <div style="color: var(--gray-700); font-size: 0.95rem;">AI-Powered Dermatological Analysis</div>
                    </div>
                </div>

                <div class="module-description">
                    Upload dermatological images for automated screening of suspicious skin lesions. Supports JPG, PNG formats. Images should be clear, well-lit, and focused on the skin lesion.
                </div>

                <!-- Skin version toggle -->
                <div class="version-toggle" id="skinVersionToggle">
                    <button class="v-btn v1-active" id="skinVBtnV1" onclick="setSkinVersion('v1')">
                        <span class="v-pip"></span> v1 &nbsp;Image-only
                    </button>
                    <button class="v-btn" id="skinVBtnV2" onclick="setSkinVersion('v2')">
                        <span class="v-pip"></span> v2 &nbsp;Multimodal
                    </button>
                </div>

                <!-- Skin v2 metadata inputs -->
                <div class="metadata-panel" id="skinMetadataPanel">
                    <div class="metadata-panel-title"><i class="fas fa-user-md"></i> Patient Risk Factors</div>
                    <div class="meta-grid">
                        <div class="meta-field">
                            <label for="skinMetaAge">Age (years)</label>
                            <input type="number" id="skinMetaAge" min="18" max="120" placeholder="e.g. 45" value="45">
                        </div>
                        <div class="meta-field">
                            <label for="skinMetaType" style="display:flex;align-items:center;gap:6px;">
                                Fitzpatrick Skin Type
                                <span class="tooltip-icon" title="1=Very Fair (always burns), 2=Fair (usually burns), 3=Medium (sometimes burns), 4=Olive (rarely burns), 5=Brown (very rarely burns), 6=Dark (never burns)">&#9432;</span>
                            </label>
                            <select id="skinMetaType">
                                <option value="1">Type I &ndash; Very Fair</option>
                                <option value="2">Type II &ndash; Fair</option>
                                <option value="3" selected>Type III &ndash; Medium</option>
                                <option value="4">Type IV &ndash; Olive</option>
                                <option value="5">Type V &ndash; Brown</option>
                                <option value="6">Type VI &ndash; Dark</option>
                            </select>
                        </div>
                        <div class="meta-field">
                            <label for="skinMetaSunburns" style="display:flex;align-items:center;gap:6px;">
                                Significant Sunburns (lifetime)
                                <span class="tooltip-icon" title="Number of significant sunburns (blistering/peeling) in your lifetime.">&#9432;</span>
                            </label>
                            <input type="number" id="skinMetaSunburns" min="0" max="50" placeholder="e.g. 3" value="2">
                        </div>
                        <div class="meta-field">
                            <label for="skinMetaOutdoorHours" style="display:flex;align-items:center;gap:6px;">
                                Outdoor Hours/Week
                                <span class="tooltip-icon" title="Average hours spent outdoors per week, year-round.">&#9432;</span>
                            </label>
                            <input type="number" id="skinMetaOutdoorHours" min="0" max="112" placeholder="e.g. 10" value="10">
                        </div>
                        <div class="meta-field">
                            <label for="skinMetaTanningBed">Tanning Bed Use</label>
                            <select id="skinMetaTanningBed">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        <div class="meta-field">
                            <label for="skinMetaFamilyHistory">Family History of Skin Cancer</label>
                            <select id="skinMetaFamilyHistory">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div class="upload-area skin" id="skinUploadArea">
                    <input type="file" id="skinFileInput" class="file-input" accept="image/*">
                    <img id="skinPreview" class="upload-preview" alt="Preview">
                    <div id="skinUploadContent">
                        <div class="upload-icon">
                            <i class="fas fa-cloud-upload-alt"></i>
                        </div>
                        <div class="upload-text">
                            <h3>Upload Skin Image</h3>
                            <p>JPG, PNG, or JPEG format (max 5MB)</p>
                            <p>Dermatological lesions only</p>
                        </div>
                    </div>
                    <div id="skinFileName" class="file-info skin" style="display: none;">
                        <i class="fas fa-file-image"></i>
                        <span></span>
                    </div>
                </div>

                <button class="analyze-btn skin" onclick="analyzeSkin()" id="skinAnalyzeBtn">
                    <i class="fas fa-play-circle"></i>
                    <span id="skinBtnText">Analyze Skin Image</span>
                </button>

                <!-- Error banner -->
                <div class="error-banner" id="skinErrorBanner">
                    <i class="fas fa-exclamation-circle"></i>
                    <div class="error-banner-body">
                        <div class="error-banner-title">Analysis Error</div>
                        <div class="error-banner-msg" id="skinErrorMsg">An error occurred.</div>
                    </div>
                    <button class="error-banner-close" onclick="hideError('skin')" title="Dismiss">&#10005;</button>
                </div>

                <div class="loading-container" id="skinLoading">
                    <div class="spinner"></div>
                    <p class="loading-text">Processing dermatological patterns<span class="loading-dots"><span></span><span></span><span></span></span></p>
                    <p class="loading-subtext">Analyzing epidermal structures and lesion morphology</p>
                </div>

                <div class="results-container skin" id="skinResults">
                    <div class="result-header">
                        <div class="result-title">
                            <i class="fas fa-hand"></i>
                            Skin Screening Results
                        </div>
                        <div class="risk-badge" id="skinRiskBadge">--</div>
                    </div>
                    <div class="result-content">
                        <div class="probability-section">
                            <div class="probability-label">
                                <span>Suspicious Pattern Probability</span>
                                <span id="skinProbabilityValue">0%</span>
                            </div>
                            <div class="probability-bar">
                                <div class="probability-fill" id="skinProbabilityFill"></div>
                            </div>
                            <div class="probability-scale">
                                <span>Benign</span>
                                <span>Atypical</span>
                                <span>Suspicious</span>
                            </div>
                        </div>

                        <div class="metrics-grid">
                            <div class="metric-box">
                                <div class="metric-value" id="skinPatternProb">--%</div>
                                <div class="metric-label">Pattern Score</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value" id="skinPrediction">--</div>
                                <div class="metric-label">Screening Result</div>
                            </div>
                        </div>

                        <div class="recommendation-box">
                            <div class="recommendation-header">
                                <i class="fas fa-clipboard-check"></i>
                                Clinical Guidance
                            </div>
                            <p id="skinRecommendation">Upload a skin image to receive screening guidance based on lesion pattern analysis.</p>
                        </div>

                        <div class="disclaimer">
                            <div class="disclaimer-header">
                                <i class="fas fa-exclamation-triangle"></i>
                                Important Notice
                            </div>
                            <p>This is an AI-assisted screening tool, NOT a diagnostic system. All results require confirmation by qualified dermatologists. Intended for screening purposes only.</p>
                        </div>
                    </div>
                </div>

                <!-- Skin v2 results container (with GradCAM) -->
                <div class="results-container skin" id="skinV2Results">
                    <div class="result-header">
                        <div class="result-title">
                            <i class="fas fa-brain"></i>
                            Multimodal Results <span class="v2-tag" style="font-size:0.7rem;">v2</span>
                        </div>
                        <div class="risk-badge filled" id="skinV2RiskBadge">--</div>
                    </div>
                    <div class="result-content">
                        <div class="probability-section">
                            <div class="probability-label">
                                <span>Malignancy Probability</span>
                                <span id="skinV2ProbValue">0%</span>
                            </div>
                            <div class="probability-bar">
                                <div class="probability-fill" id="skinV2ProbFill"></div>
                            </div>
                            <div class="probability-scale">
                                <span>Low Risk</span>
                                <span>Medium Risk</span>
                                <span>High Risk</span>
                            </div>
                        </div>

                        <!-- Metadata chips -->
                        <div class="meta-result-row" id="skinV2MetaChips"></div>

                        <div class="metrics-grid">
                            <div class="metric-box">
                                <div class="metric-value" id="skinV2PatternProb">--%</div>
                                <div class="metric-label">Pattern Score</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value" id="skinV2RiskLabel" style="font-size:1.4rem;">--</div>
                                <div class="metric-label">Risk Tier</div>
                            </div>
                        </div>

                        <!-- GradCAM side-by-side panel -->
                        <div class="gradcam-panel" id="skinGradcamPanel" style="display:none;">
                            <div class="gradcam-title">
                                <i class="fas fa-eye"></i> Grad-CAM Explainability
                                <span class="tooltip-icon" title="Gradient-weighted Class Activation Map highlights which image regions most influenced the model's prediction. Red/hot areas have the highest influence.">&#9432;</span>
                            </div>
                            <div class="gradcam-opacity-row">
                                <label for="skinGradcamOpacity" style="font-size:0.82rem;color:var(--gray-600);">Heatmap Opacity</label>
                                <input type="range" id="skinGradcamOpacity" min="0" max="100" value="70"
                                       oninput="updateSkinGradcamOpacity(this.value)"
                                       style="flex:1;accent-color:var(--primary);">
                                <span id="skinGradcamOpacityVal" style="font-size:0.82rem;color:var(--gray-600);min-width:32px;text-align:right;">70%</span>
                            </div>
                            <div class="gradcam-images">
                                <div class="gradcam-image-box">
                                    <img id="skinOriginalImg" alt="Original Image" style="display:none;">
                                    <div class="gradcam-image-label">Original Image</div>
                                </div>
                                <div class="gradcam-image-box" style="position:relative;">
                                    <img id="skinGradcamImg" alt="Grad-CAM Overlay" style="opacity:0.7;display:none;">
                                    <div class="gradcam-image-label">&#128293; Activation Heatmap</div>
                                </div>
                            </div>
                        </div>

                        <div class="recommendation-box">
                            <div class="recommendation-header">
                                <i class="fas fa-clipboard-check"></i>
                                Clinical Guidance
                            </div>
                            <p id="skinV2Recommendation">Upload a skin image to receive screening guidance.</p>
                        </div>

                        <div class="clinical-interpretation" id="skinV2ClinicalInterp" style="display:none;">
                            <div class="clinical-interp-header">
                                <i class="fas fa-stethoscope"></i> AI Clinical Interpretation
                            </div>
                            <p id="skinV2InterpText"></p>
                        </div>

                        <div class="meta-warnings" id="skinV2MetaWarnings" style="display:none;">
                            <div style="font-size:0.78rem;font-weight:600;color:var(--warning);margin-bottom:4px;">&#9888;&#65039; Input Notes</div>
                            <ul id="skinV2WarningsList" style="font-size:0.78rem;color:var(--gray-600);padding-left:1.2em;margin:0;"></ul>
                        </div>

                        <div class="disclaimer">
                            <div class="disclaimer-header">
                                <i class="fas fa-exclamation-triangle"></i>
                                Important Notice
                            </div>
                            <p>Experimental v2 multimodal model &mdash; not validated for clinical use. Always consult a qualified dermatologist.</p>
                        </div>
                    </div>
                </div>

                <div class="placeholder" id="skinPlaceholder">
                    <i class="fas fa-chart-line"></i>
                    <h3 style="margin-bottom: 10px;">Awaiting Skin Analysis</h3>
                    <p>Upload a dermatological image to begin lesion pattern screening</p>
                </div>
            </div>

            <!-- ══════════════════════════════════════════════════════
                 FIELD SCREENING MODULE  (for health-camp workers)
                 ══════════════════════════════════════════════════════ -->
            <div class="module-card field">
                <div class="module-header">
                    <div class="module-icon field">
                        <i class="fas fa-user-injured"></i>
                    </div>
                    <div>
                        <div class="module-title">🩺 Field Screening &amp; Report</div>
                        <div style="color: var(--gray-700); font-size: 0.95rem;">Patient Record + Printable PDF Report</div>
                    </div>
                </div>

                <div class="module-description">
                    For health workers at camps. Fill patient details, upload a photo, and download a printable report to share with doctors.
                </div>

                <!-- ── Patient info form ── -->
                <div class="field-form-grid">
                    <div class="meta-field">
                        <label>Patient Name *</label>
                        <input type="text" id="fieldName" placeholder="Full name">
                    </div>
                    <div class="meta-field">
                        <label>Age (years) *</label>
                        <input type="number" id="fieldAge" min="1" max="120" placeholder="e.g. 45">
                    </div>
                    <div class="meta-field">
                        <label>Gender</label>
                        <select id="fieldGender">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>
                    <div class="meta-field">
                        <label>Village / Area</label>
                        <input type="text" id="fieldVillage" placeholder="e.g. Nandurbar">
                    </div>
                    <div class="meta-field">
                        <label>Contact Number</label>
                        <input type="tel" id="fieldContact" placeholder="10-digit number">
                    </div>
                    <div class="meta-field">
                        <label>Screened By</label>
                        <input type="text" id="fieldScreenedBy" placeholder="Health worker name">
                    </div>
                    <div class="meta-field" style="grid-column:1/-1;">
                        <label>Screening Type</label>
                        <select id="fieldScreeningType">
                            <option value="oral">Oral Cancer Screening</option>
                            <option value="skin">Skin Cancer Screening</option>
                        </select>
                    </div>
                </div>

                <!-- ── Image upload ── -->
                <div class="upload-area field" id="fieldUploadArea">
                    <input type="file" id="fieldFileInput" class="file-input" accept="image/*">
                    <img id="fieldPreview" class="upload-preview" alt="Preview">
                    <div id="fieldUploadContent">
                        <div class="upload-icon">
                            <i class="fas fa-camera"></i>
                        </div>
                        <div class="upload-text">
                            <h3>Upload Patient Photo</h3>
                            <p>Take a clear photo of the affected area</p>
                        </div>
                    </div>
                    <div id="fieldFileName" class="file-info" style="display: none;">
                        <i class="fas fa-file-image"></i>
                        <span></span>
                    </div>
                </div>

                <!-- ── Analyze button ── -->
                <button class="analyze-btn field" onclick="analyzeField()" id="fieldAnalyzeBtn">
                    <i class="fas fa-stethoscope"></i>
                    <span id="fieldBtnText">Screen Patient</span>
                </button>

                <!-- ── Error banner ── -->
                <div class="error-banner" id="fieldErrorBanner">
                    <i class="fas fa-exclamation-circle"></i>
                    <div class="error-banner-body">
                        <div class="error-banner-title">Screening Error</div>
                        <div class="error-banner-msg" id="fieldErrorMsg">An error occurred.</div>
                    </div>
                    <button class="error-banner-close" onclick="hideError('field')" title="Dismiss">&#10005;</button>
                </div>

                <!-- ── Loading ── -->
                <div class="loading-container" id="fieldLoading" style="display:none;">
                    <div class="spinner"></div>
                    <p class="loading-text">Analyzing<span class="loading-dots"><span></span><span></span><span></span></span></p>
                    <p class="loading-subtext">Running AI model, please wait…</p>
                </div>

                <!-- ── Results ── -->
                <div id="fieldResults" style="display:none;">

                    <!-- Big colour-coded risk card -->
                    <div class="field-result-card" id="fieldRiskCard">
                        <div class="field-risk-icon" id="fieldRiskIcon">⚠️</div>
                        <div class="field-risk-label" id="fieldRiskLabel">--</div>
                        <div class="field-risk-action" id="fieldRiskAction">--</div>
                        <div class="field-prob-text" id="fieldProbText"></div>
                    </div>

                    <!-- Download report -->
                    <a id="fieldReportBtn" href="#" target="_blank" class="report-download-btn" style="display:none;">
                        <i class="fas fa-file-pdf"></i>
                        Download PDF Report
                    </a>

                    <div class="disclaimer">
                        <div class="disclaimer-header">
                            <i class="fas fa-exclamation-triangle"></i>
                            Important Notice
                        </div>
                        <p>This is an AI-assisted screening tool only. Refer all high-risk patients to a qualified doctor immediately. Record must be confirmed by a medical professional.</p>
                    </div>
                </div>

                <!-- ── Placeholder ── -->
                <div class="placeholder" id="fieldPlaceholder">
                    <i class="fas fa-notes-medical"></i>
                    <h3 style="margin-bottom: 10px;">Ready to Screen</h3>
                    <p>Fill patient details and upload a photo to begin</p>
                </div>

                <!-- ── Patient history ── -->
                <button class="history-toggle-btn" onclick="toggleHistory()">
                    <i class="fas fa-history"></i> View Recent Screenings
                </button>
                <div class="history-table-wrap" id="historyTableWrap">
                    <table id="historyTable">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Name</th>
                                <th>Age</th>
                                <th>Village</th>
                                <th>Type</th>
                                <th>Risk</th>
                                <th>Report</th>
                            </tr>
                        </thead>
                        <tbody id="historyTableBody">
                            <tr><td colspan="7" style="text-align:center;color:var(--gray-500);padding:12px;">Loading…</td></tr>
                        </tbody>
                    </table>
                </div>

            </div>
        </div>

        <div class="platform-stats">
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-chart-line"></i></div>
                <div class="stat-value">{{ auc_score }}</div>
                <div class="stat-label">Model AUC Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-search-plus"></i></div>
                <div class="stat-value">{{ sensitivity }}%</div>
                <div class="stat-label">Pattern Sensitivity</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-shield-alt"></i></div>
                <div class="stat-value">{{ specificity }}%</div>
                <div class="stat-label">Pattern Specificity</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-sliders-h"></i></div>
                <div class="stat-value">{{ optimal_threshold }}</div>
                <div class="stat-label">Optimal Threshold</div>
            </div>
        </div>

        <footer>
            <p>CuraLens AI Screening Platform v2.1 | Dual-Module Architecture | Research Use Only | Not for Diagnostic Purposes</p>
            <p style="margin-top: 5px; font-size: 0.85rem;">© 2024 AI-Assisted Medical Screening Research</p>
            <div class="footer-badge">
                <i class="fas fa-flask"></i>
                Educational & Research Purpose Only
            </div>
        </footer>
    </div>

    <script>
        let oralFile  = null;
        let skinFile  = null;
        let fieldFile = null;
        let currentVersion     = 'v1';   // oral: 'v1' | 'v2'
        let skinCurrentVersion = 'v1';   // skin: 'v1' | 'v2'

        function setupUpload(uploadAreaId, fileInputId, fileNameId, previewId, uploadContentId, fileVarName) {
            const uploadArea = document.getElementById(uploadAreaId);
            const fileInput = document.getElementById(fileInputId);
            const fileName = document.getElementById(fileNameId);
            const preview = document.getElementById(previewId);
            const uploadContent = document.getElementById(uploadContentId);

            // Open file dialog only when clicking the upload area itself,
            // NOT when clicking the preview image (which would re-open the picker)
            uploadArea.addEventListener('click', (e) => {
                if (e.target === preview) return;   // ignore clicks ON the preview img
                fileInput.click();
            });

            // Prevent the preview img from bubbling its own click to the area
            preview.addEventListener('click', (e) => e.stopPropagation());

            ['dragenter', 'dragover'].forEach(event => {
                uploadArea.addEventListener(event, (e) => {
                    e.preventDefault();
                    uploadArea.classList.add('dragover');
                });
            });

            ['dragleave', 'drop'].forEach(event => {
                uploadArea.addEventListener(event, (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                });
            });

            uploadArea.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                handleFileSelect(files, fileName, preview, uploadContent, uploadArea, fileVarName);
            });

            fileInput.addEventListener('change', (e) => {
                handleFileSelect(e.target.files, fileName, preview, uploadContent, uploadArea, fileVarName);
            });
        }

        function handleFileSelect(files, fileNameElement, previewElement, uploadContent, uploadArea, fileVarName) {
            if (files.length > 0) {
                const file = files[0];
                if (fileVarName === 'oralFile')  oralFile  = file;
                if (fileVarName === 'skinFile')  skinFile  = file;
                if (fileVarName === 'fieldFile') fieldFile = file;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewElement.src = e.target.result;
                    previewElement.style.display = 'block';  // ensure visible
                    previewElement.classList.add('visible');
                    uploadContent.style.display = 'none';
                    uploadArea.classList.add('has-image');
                };
                reader.onerror = function() {
                    console.warn('FileReader error – could not read selected file.');
                };
                reader.readAsDataURL(file);
                
                // Update file name display
                fileNameElement.querySelector('span').textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
                fileNameElement.style.display = 'inline-flex';
            }
        }

        setupUpload('oralUploadArea',  'oralFileInput',  'oralFileName',  'oralPreview',  'oralUploadContent',  'oralFile');
        setupUpload('skinUploadArea',  'skinFileInput',  'skinFileName',  'skinPreview',  'skinUploadContent',  'skinFile');
        setupUpload('fieldUploadArea', 'fieldFileInput', 'fieldFileName', 'fieldPreview', 'fieldUploadContent', 'fieldFile');

        // ── Oral version toggle ───────────────────────────────────────
        function setVersion(v) {
            currentVersion = v;
            document.getElementById('vBtnV1').className = 'v-btn' + (v === 'v1' ? ' v1-active' : '');
            document.getElementById('vBtnV2').className = 'v-btn' + (v === 'v2' ? ' v2-active' : '');
            document.getElementById('metadataPanel').classList.toggle('visible', v === 'v2');
            document.getElementById('oralBtnText').textContent =
                v === 'v2' ? 'Analyze with Multimodal v2' : 'Analyze Oral Image';
            // Reset results when switching version
            ['oralResults','oralV2Results'].forEach(id => {
                document.getElementById(id).style.display = 'none';
            });
            document.getElementById('oralPlaceholder').style.display = 'block';
            hideError('oral');
        }

        // ── Skin version toggle ───────────────────────────────────────
        function setSkinVersion(v) {
            skinCurrentVersion = v;
            document.getElementById('skinVBtnV1').className = 'v-btn' + (v === 'v1' ? ' v1-active' : '');
            document.getElementById('skinVBtnV2').className = 'v-btn' + (v === 'v2' ? ' v2-active' : '');
            document.getElementById('skinMetadataPanel').classList.toggle('visible', v === 'v2');
            document.getElementById('skinBtnText').textContent =
                v === 'v2' ? 'Analyze with Multimodal v2' : 'Analyze Skin Image';
            // Reset results when switching version
            ['skinResults','skinV2Results'].forEach(id => {
                document.getElementById(id).style.display = 'none';
            });
            document.getElementById('skinPlaceholder').style.display = 'block';
            hideError('skin');
        }

        // ── Error banner helpers ──────────────────────────────────────
        function showError(type, msg) {
            const banner = document.getElementById(type + 'ErrorBanner');
            const msgEl  = document.getElementById(type + 'ErrorMsg');
            if (!banner) return;
            if (msgEl) msgEl.textContent = msg;
            banner.classList.add('visible');
        }
        function hideError(type) {
            const banner = document.getElementById(type + 'ErrorBanner');
            if (banner) banner.classList.remove('visible');
        }

        // ── Risk badge colour helper ──────────────────────────────────
        function applyRiskBadge(badgeEl, levelStr) {
            // levelStr: e.g. "High", "HIGH", "MODERATE", "Low", "VERY LOW"
            const key = levelStr.toLowerCase().replace(/[\s_-]/g, '');
            // normalise to simple tier
            let tier = key;
            if (key === 'verylow') tier = 'very-low';
            else if (key === 'moderate') tier = 'moderate';
            badgeEl.textContent  = levelStr.toUpperCase();
            badgeEl.className    = 'risk-badge filled ' + tier;
        }

        // ── Oral analysis dispatcher ──────────────────────────────────
        async function analyzeOral() {
            if (!oralFile) {
                showError('oral', 'Please select an oral image before running analysis.');
                return;
            }
            hideError('oral');
            if (currentVersion === 'v2') {
                await analyzeOralV2();
            } else {
                await analyzeOralV1();
            }
        }

        async function analyzeOralV1() {
            const formData = new FormData();
            formData.append('image', oralFile);

            showLoading('oral');
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                updateOralResults(data);
            } catch (error) {
                showError('oral', error.message);
            } finally {
                hideLoading('oral');
            }
        }

        /* ── Cancer-type dropdown handler ───────────────── */
        function onCancerTypeChange(value) {
            ['metaGroupOralLegacy', 'metaGroupOral', 'metaGroupSkin'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = 'none';
            });
            const map = {
                oral_legacy: 'metaGroupOralLegacy',
                oral:        'metaGroupOral',
                skin:        'metaGroupSkin'
            };
            const target = document.getElementById(map[value]);
            if (target) target.style.display = 'block';
        }

        /* ── Grad-CAM opacity sliders ────────────────────── */
        function updateGradcamOpacity(val) {
            const img = document.getElementById('oralGradcamImg');
            if (img) img.style.opacity = (val / 100).toFixed(2);
            const lbl = document.getElementById('gradcamOpacityVal');
            if (lbl) lbl.textContent = val + '%';
        }
        function updateSkinGradcamOpacity(val) {
            const img = document.getElementById('skinGradcamImg');
            if (img) img.style.opacity = (val / 100).toFixed(2);
            const lbl = document.getElementById('skinGradcamOpacityVal');
            if (lbl) lbl.textContent = val + '%';
        }

        /* ── V2 analysis (multi-modality aware) ──────────── */
        async function analyzeOralV2() {
            const formData = new FormData();
            formData.append('image', oralFile);

            const cancerType = (document.getElementById('metaCancerType') || {}).value || 'oral_legacy';
            formData.append('cancer_type', cancerType);

            if (cancerType === 'oral_legacy') {
                formData.append('age',          document.getElementById('metaAge')?.value         || '0');
                formData.append('smoking',       document.getElementById('metaSmoking')?.value     || '0');
                formData.append('alcohol',       document.getElementById('metaAlcohol')?.value     || '0');
                formData.append('sun_exposure',  document.getElementById('metaSunExposure')?.value || '0');

            } else if (cancerType === 'oral') {
                formData.append('age',                       document.getElementById('oralAge')?.value             || '0');
                formData.append('smoking_years',             document.getElementById('oralSmokingYears')?.value     || '0');
                formData.append('cigarettes_per_day',        document.getElementById('oralCigsPerDay')?.value       || '0');
                formData.append('alcohol_units_per_week',    document.getElementById('oralAlcoholUnits')?.value     || '0');
                formData.append('chewing_tobacco',           document.getElementById('oralChewingTobacco')?.value   || '0');
                formData.append('family_history',            document.getElementById('oralFamilyHistory')?.value    || '0');

            } else if (cancerType === 'skin') {
                formData.append('age',                       document.getElementById('skinAge')?.value              || '0');
                formData.append('skin_type',                 document.getElementById('skinType')?.value             || '3');
                formData.append('sunburn_history',           document.getElementById('skinSunburns')?.value         || '0');
                formData.append('outdoor_hours_per_week',    document.getElementById('skinOutdoorHours')?.value     || '0');
                formData.append('tanning_bed_use',           document.getElementById('skinTanningBed')?.value       || '0');
                formData.append('family_history',            document.getElementById('skinFamilyHistory')?.value    || '0');
            }

            showLoading('oral');
            try {
                const response = await fetch('/predict_v2', { method: 'POST', body: formData });
                const data = await response.json();
                if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
                updateOralV2Results(data);
            } catch (error) {
                showError('oral', error.message);
            } finally {
                hideLoading('oral');
            }
        }

        async function analyzeSkin() {
            if (!skinFile) {
                showError('skin', 'Please select a skin image before running analysis.');
                return;
            }
            hideError('skin');
            if (skinCurrentVersion === 'v2') {
                await analyzeSkinV2();
            } else {
                await analyzeSkinV1();
            }
        }

        async function analyzeSkinV1() {
            const formData = new FormData();
            formData.append('image', skinFile);

            showLoading('skin');
            try {
                const response = await fetch('/predict/skin', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                updateSkinResults(data);
            } catch (error) {
                showError('skin', error.message);
            } finally {
                hideLoading('skin');
            }
        }

        async function analyzeSkinV2() {
            const formData = new FormData();
            formData.append('image', skinFile);
            formData.append('cancer_type', 'skin');
            formData.append('age',                    document.getElementById('skinMetaAge')?.value          || '45');
            formData.append('skin_type',              document.getElementById('skinMetaType')?.value         || '3');
            formData.append('sunburn_history',        document.getElementById('skinMetaSunburns')?.value     || '0');
            formData.append('outdoor_hours_per_week', document.getElementById('skinMetaOutdoorHours')?.value || '0');
            formData.append('tanning_bed_use',        document.getElementById('skinMetaTanningBed')?.value   || '0');
            formData.append('family_history',         document.getElementById('skinMetaFamilyHistory')?.value|| '0');

            showLoading('skin');
            try {
                const response = await fetch('/predict_v2', { method: 'POST', body: formData });
                const data = await response.json();
                if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
                updateSkinV2Results(data);
            } catch (error) {
                showError('skin', error.message);
            } finally {
                hideLoading('skin');
            }
        }

        function showLoading(type) {
            document.getElementById(`${type}Loading`).style.display = 'block';
            const btn = document.getElementById(`${type}AnalyzeBtn`);
            btn.disabled = true;
            // Show spinner in button text if possible
            const btnTxt = document.getElementById(`${type}BtnText`);
            if (btnTxt) { btn._prevText = btnTxt.textContent; btnTxt.textContent = 'Analyzing\u2026'; }
            document.getElementById(`${type}Placeholder`).style.display = 'none';
            document.getElementById(`${type}Results`).style.display = 'none';
            const v2Res = document.getElementById(`${type}V2Results`);
            if (v2Res) v2Res.style.display = 'none';
        }

        function hideLoading(type) {
            document.getElementById(`${type}Loading`).style.display = 'none';
            const btn = document.getElementById(`${type}AnalyzeBtn`);
            btn.disabled = false;
            const btnTxt = document.getElementById(`${type}BtnText`);
            if (btnTxt && btn._prevText) btnTxt.textContent = btn._prevText;
        }

        function updateOralResults(data) {
            const prob = (data.cancer_probability * 100).toFixed(1);
            let riskLevel = 'low';
            if (data.risk_level) riskLevel = data.risk_level.toLowerCase();
            else if (prob > 70) riskLevel = 'high';
            else if (prob > 40) riskLevel = 'moderate';

            document.getElementById('oralPatternProb').textContent    = prob + '%';
            document.getElementById('oralPrediction').textContent      = data.prediction || 'Pattern Analyzed';
            document.getElementById('oralProbabilityValue').textContent = prob + '%';
            document.getElementById('oralRecommendation').textContent  = data.recommendation || 'Further clinical evaluation recommended.';
            document.getElementById('oralProbabilityFill').style.width = prob + '%';

            applyRiskBadge(document.getElementById('oralRiskBadge'), riskLevel);

            document.getElementById('oralPlaceholder').style.display = 'none';
            document.getElementById('oralResults').style.display = 'block';
        }

        function updateOralV2Results(data) {
            const prob    = parseFloat(data.probability || 0);
            const probPct = (prob * 100).toFixed(1);

            document.getElementById('oralV2PatternProb').textContent    = probPct + '%';
            document.getElementById('oralV2ProbValue').textContent      = probPct + '%';
            document.getElementById('oralV2ProbFill').style.width       = probPct + '%';
            document.getElementById('oralV2RiskLabel').textContent      = data.risk_label || data.risk_level || '--';
            document.getElementById('oralV2Recommendation').textContent = data.recommendation || 'Consult a healthcare professional.';

            applyRiskBadge(document.getElementById('oralV2RiskBadge'), data.risk_level || 'low');

            // ── Metadata chips (cancer-type aware) ──
            const meta       = data.metadata_used || {};
            const cancerType = data.cancer_type   || 'oral_legacy';
            const chips      = document.getElementById('oralV2MetaChips');
            let chipHtml     = '';
            if (cancerType === 'oral_legacy') {
                chipHtml = [
                    meta.age         != null ? `<span class="meta-result-chip"><i class="fas fa-user"></i> Age: ${meta.age}</span>` : '',
                    meta.smoking     != null ? `<span class="meta-result-chip"><i class="fas fa-smoking${meta.smoking ? '' : '-ban'}"></i> ${meta.smoking ? 'Smoker' : 'Non-smoker'}</span>` : '',
                    meta.alcohol     != null ? `<span class="meta-result-chip"><i class="fas fa-wine-glass"></i> ${meta.alcohol ? 'Drinker' : 'Non-drinker'}</span>` : '',
                    meta.sun_exposure!= null ? `<span class="meta-result-chip"><i class="fas fa-sun"></i> Sun: ${meta.sun_exposure}</span>` : '',
                ].join('');
            } else if (cancerType === 'oral') {
                chipHtml = [
                    meta.age                    != null ? `<span class="meta-result-chip"><i class="fas fa-user"></i> Age: ${meta.age}</span>` : '',
                    meta.smoking_years          != null ? `<span class="meta-result-chip"><i class="fas fa-smoking"></i> Smoked: ${meta.smoking_years} yr</span>` : '',
                    meta.cigarettes_per_day     != null ? `<span class="meta-result-chip"><i class="fas fa-fire"></i> ${meta.cigarettes_per_day} cig/day</span>` : '',
                    meta.alcohol_units_per_week != null ? `<span class="meta-result-chip"><i class="fas fa-wine-glass"></i> ${meta.alcohol_units_per_week} u/wk</span>` : '',
                    meta.chewing_tobacco        != null ? `<span class="meta-result-chip"><i class="fas fa-leaf"></i> Chew: ${meta.chewing_tobacco ? 'Yes' : 'No'}</span>` : '',
                    meta.family_history         != null ? `<span class="meta-result-chip"><i class="fas fa-dna"></i> Fam Hx: ${meta.family_history ? 'Yes' : 'No'}</span>` : '',
                ].join('');
            } else if (cancerType === 'skin') {
                chipHtml = [
                    meta.age                   != null ? `<span class="meta-result-chip"><i class="fas fa-user"></i> Age: ${meta.age}</span>` : '',
                    meta.skin_type             != null ? `<span class="meta-result-chip"><i class="fas fa-palette"></i> Fitzpatrick: ${meta.skin_type}</span>` : '',
                    meta.sunburn_history       != null ? `<span class="meta-result-chip"><i class="fas fa-sun"></i> Burns: ${meta.sunburn_history}</span>` : '',
                    meta.outdoor_hours_per_week!= null ? `<span class="meta-result-chip"><i class="fas fa-walking"></i> Outdoor: ${meta.outdoor_hours_per_week} hr/wk</span>` : '',
                    meta.tanning_bed_use       != null ? `<span class="meta-result-chip"><i class="fas fa-bed"></i> Tan Bed: ${meta.tanning_bed_use ? 'Yes' : 'No'}</span>` : '',
                    meta.family_history        != null ? `<span class="meta-result-chip"><i class="fas fa-dna"></i> Fam Hx: ${meta.family_history ? 'Yes' : 'No'}</span>` : '',
                ].join('');
            }
            chips.innerHTML = chipHtml;

            // ── Clinical interpretation ──
            const interpSec  = document.getElementById('oralV2ClinicalInterp');
            const interpText = document.getElementById('oralV2InterpText');
            if (interpSec && interpText) {
                let interp = data.clinical_interpretation || '';
                if (!interp) {
                    const rl = (data.risk_level || '').toLowerCase();
                    const ct = cancerType === 'skin' ? 'skin' : 'oral';
                    if (rl === 'high') {
                        interp = ct === 'skin'
                            ? 'The image and clinical data indicate high-risk features. Urgent dermatological evaluation and possible biopsy (e.g., punch or shave) are recommended. Do not delay follow-up.'
                            : 'The image and clinical factors suggest significant risk. An urgent referral to an oral and maxillofacial specialist or oncologist for biopsy is strongly recommended.';
                    } else if (rl === 'medium' || rl === 'moderate') {
                        interp = ct === 'skin'
                            ? 'Moderate risk features identified. Schedule a dermatology review within 4\u20136 weeks. Monitor any changing lesions and apply sun-safe behaviours.'
                            : 'Moderate clinical risk detected. Schedule a dental or oral surgery consultation within 4\u20136 weeks and avoid known risk factors (tobacco, alcohol).';
                    } else {
                        interp = ct === 'skin'
                            ? 'Low-risk features on current assessment. Continue routine annual skin checks. Use SPF 50+ sunscreen daily and perform monthly self-examinations.'
                            : 'Low-risk features on current assessment. Maintain routine dental check-ups every 6 months and follow healthy lifestyle practices to minimise future risk.';
                    }
                }
                interpText.textContent = interp;
                interpSec.style.display = 'block';
            }

            // ── Metadata warnings ──
            const warnSec  = document.getElementById('oralV2MetaWarnings');
            const warnList = document.getElementById('oralV2WarningsList');
            if (warnSec && warnList && data.metadata_warnings && data.metadata_warnings.length) {
                warnList.innerHTML = data.metadata_warnings.map(w =>
                    `<li style="font-size:0.84rem;color:var(--gray-700);margin-bottom:4px;">${w}</li>`
                ).join('');
                warnSec.style.display = 'block';
            } else if (warnSec) {
                warnSec.style.display = 'none';
            }

            // ── Grad-CAM ──
            const gcPanel = document.getElementById('oralGradcamPanel');
            if (data.gradcam_png_b64) {
                const gcImg   = document.getElementById('oralGradcamImg');
                const origImg = document.getElementById('oralOriginalImg');
                const prevImg = document.getElementById('oralPreview');
                gcImg.src   = 'data:image/png;base64,' + data.gradcam_png_b64;
                gcImg.style.display = 'block';
                if (prevImg && prevImg.src && !prevImg.src.endsWith(window.location.href)) {
                    origImg.src = prevImg.src;
                    origImg.style.display = 'block';
                }
                gcPanel.style.display = 'block';
            } else {
                gcPanel.style.display = 'none';
            }

            document.getElementById('oralPlaceholder').style.display  = 'none';
            document.getElementById('oralResults').style.display      = 'none';
            document.getElementById('oralV2Results').style.display    = 'block';
        }

        function updateSkinV2Results(data) {
            const prob    = parseFloat(data.probability || 0);
            const probPct = (prob * 100).toFixed(1);

            document.getElementById('skinV2PatternProb').textContent    = probPct + '%';
            document.getElementById('skinV2ProbValue').textContent      = probPct + '%';
            document.getElementById('skinV2ProbFill').style.width       = probPct + '%';
            document.getElementById('skinV2RiskLabel').textContent      = data.risk_label || data.risk_level || '--';
            document.getElementById('skinV2Recommendation').textContent = data.recommendation || 'Consult a dermatologist.';

            applyRiskBadge(document.getElementById('skinV2RiskBadge'), data.risk_level || 'low');

            // ── Metadata chips ──
            const meta  = data.metadata_used || {};
            const chips = document.getElementById('skinV2MetaChips');
            chips.innerHTML = [
                meta.age                   != null ? `<span class="meta-result-chip"><i class="fas fa-user"></i> Age: ${meta.age}</span>` : '',
                meta.skin_type             != null ? `<span class="meta-result-chip"><i class="fas fa-palette"></i> Fitzpatrick: ${meta.skin_type}</span>` : '',
                meta.sunburn_history       != null ? `<span class="meta-result-chip"><i class="fas fa-sun"></i> Burns: ${meta.sunburn_history}</span>` : '',
                meta.outdoor_hours_per_week!= null ? `<span class="meta-result-chip"><i class="fas fa-walking"></i> Outdoor: ${meta.outdoor_hours_per_week} hr/wk</span>` : '',
                meta.tanning_bed_use       != null ? `<span class="meta-result-chip"><i class="fas fa-bed"></i> Tan Bed: ${meta.tanning_bed_use ? 'Yes' : 'No'}</span>` : '',
                meta.family_history        != null ? `<span class="meta-result-chip"><i class="fas fa-dna"></i> Fam Hx: ${meta.family_history ? 'Yes' : 'No'}</span>` : '',
            ].join('');

            // ── Clinical interpretation ──
            const interpSec  = document.getElementById('skinV2ClinicalInterp');
            const interpText = document.getElementById('skinV2InterpText');
            if (interpSec && interpText) {
                let interp = data.clinical_interpretation || '';
                if (!interp) {
                    const rl = (data.risk_level || '').toLowerCase();
                    if (rl === 'high') {
                        interp = 'The image and clinical data indicate high-risk features. Urgent dermatological evaluation and possible biopsy (e.g., punch or shave) are recommended. Do not delay follow-up.';
                    } else if (rl === 'medium' || rl === 'moderate') {
                        interp = 'Moderate risk features identified. Schedule a dermatology review within 4\u20136 weeks. Monitor any changing lesions and apply sun-safe behaviours.';
                    } else {
                        interp = 'Low-risk features on current assessment. Continue routine annual skin checks. Use SPF 50+ sunscreen daily and perform monthly self-examinations.';
                    }
                }
                interpText.textContent = interp;
                interpSec.style.display = 'block';
            }

            // ── Metadata warnings ──
            const warnSec  = document.getElementById('skinV2MetaWarnings');
            const warnList = document.getElementById('skinV2WarningsList');
            if (warnSec && warnList && data.metadata_warnings && data.metadata_warnings.length) {
                warnList.innerHTML = data.metadata_warnings.map(w =>
                    `<li style="font-size:0.84rem;color:var(--gray-700);margin-bottom:4px;">${w}</li>`
                ).join('');
                warnSec.style.display = 'block';
            } else if (warnSec) {
                warnSec.style.display = 'none';
            }

            // ── Grad-CAM ──
            const gcPanel = document.getElementById('skinGradcamPanel');
            if (data.gradcam_png_b64) {
                const gcImg   = document.getElementById('skinGradcamImg');
                const origImg = document.getElementById('skinOriginalImg');
                const prevImg = document.getElementById('skinPreview');
                gcImg.src = 'data:image/png;base64,' + data.gradcam_png_b64;
                gcImg.style.display = 'block';
                if (prevImg && prevImg.src && !prevImg.src.endsWith(window.location.href)) {
                    origImg.src = prevImg.src;
                    origImg.style.display = 'block';
                }
                gcPanel.style.display = 'block';
            } else {
                gcPanel.style.display = 'none';
            }

            document.getElementById('skinPlaceholder').style.display  = 'none';
            document.getElementById('skinResults').style.display      = 'none';
            document.getElementById('skinV2Results').style.display    = 'block';
        }

        function updateSkinResults(data) {
            let prob = 0;
            let riskLevel = 'low';

            if      (data.risk_score        !== undefined) prob = (parseFloat(data.risk_score)        * 100).toFixed(1);
            else if (data.probability       !== undefined) prob = (parseFloat(data.probability)       * 100).toFixed(1);
            else if (data.cancer_probability!== undefined) prob = (parseFloat(data.cancer_probability)* 100).toFixed(1);
            else if (data.confidence        !== undefined) prob = (parseFloat(data.confidence)        * 100).toFixed(1);

            if      (data.risk_level)   riskLevel = data.risk_level.toLowerCase();
            else if (prob > 70)         riskLevel = 'high';
            else if (prob > 40)         riskLevel = 'moderate';

            document.getElementById('skinPatternProb').textContent     = prob + '%';
            document.getElementById('skinPrediction').textContent      = data.prediction || 'Pattern Analyzed';
            document.getElementById('skinProbabilityValue').textContent = prob + '%';
            document.getElementById('skinRecommendation').textContent  =
                data.recommendation || 'Consult dermatologist for complete evaluation.';
            document.getElementById('skinProbabilityFill').style.width = prob + '%';

            applyRiskBadge(document.getElementById('skinRiskBadge'), riskLevel);

            document.getElementById('skinPlaceholder').style.display = 'none';
            document.getElementById('skinResults').style.display     = 'block';
        }

        window.addEventListener('DOMContentLoaded', () => {
            document.querySelectorAll('.probability-fill').forEach(fill => {
                fill.style.width = '0%';
            });
        });

        // ══════════════════════════════════════════════════════════════
        //  FIELD SCREENING MODULE — JavaScript
        // ══════════════════════════════════════════════════════════════

        async function analyzeField() {
            const name = document.getElementById('fieldName').value.trim();
            const age  = document.getElementById('fieldAge').value.trim();
            if (!name) { showError('field', 'Please enter the patient name.'); return; }
            if (!age)  { showError('field', 'Please enter the patient age.'); return; }
            if (!fieldFile) { showError('field', 'Please select a patient photo before screening.'); return; }

            hideError('field');
            document.getElementById('fieldLoading').style.display     = 'block';
            document.getElementById('fieldPlaceholder').style.display = 'none';
            document.getElementById('fieldResults').style.display     = 'none';
            const btn = document.getElementById('fieldAnalyzeBtn');
            btn.disabled = true;
            document.getElementById('fieldBtnText').textContent = 'Analyzing\u2026';

            try {
                const fd = new FormData();
                fd.append('image',         fieldFile);
                fd.append('name',          name);
                fd.append('age',           age);
                fd.append('gender',        document.getElementById('fieldGender').value);
                fd.append('village',       document.getElementById('fieldVillage').value.trim());
                fd.append('contact',       document.getElementById('fieldContact').value.trim());
                fd.append('screened_by',   document.getElementById('fieldScreenedBy').value.trim());
                fd.append('screening_type',document.getElementById('fieldScreeningType').value);

                const resp = await fetch('/screen', { method: 'POST', body: fd });
                const data = await resp.json();
                if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);
                updateFieldResults(data);
            } catch (err) {
                showError('field', err.message);
            } finally {
                document.getElementById('fieldLoading').style.display = 'none';
                btn.disabled = false;
                document.getElementById('fieldBtnText').textContent = 'Screen Patient';
            }
        }

        function updateFieldResults(data) {
            const prob      = ((data.probability || 0) * 100).toFixed(1);
            const risk      = (data.risk_level || 'low').toLowerCase();
            const patientId = data.patient_id || '';

            // Choose tier
            let tierClass = 'risk-low', icon = '✅', label = 'LOW RISK', action = 'MONITOR — No immediate action';
            if (risk === 'high') {
                tierClass = 'risk-high'; icon = '🚨'; label = 'HIGH RISK'; action = 'REFER TO DOCTOR URGENTLY';
            } else if (risk === 'medium' || risk === 'moderate') {
                tierClass = 'risk-medium'; icon = '⚠️'; label = 'MEDIUM RISK'; action = 'SCHEDULE DOCTOR VISIT SOON';
            }

            const card   = document.getElementById('fieldRiskCard');
            card.className = 'field-result-card ' + tierClass;
            document.getElementById('fieldRiskIcon').textContent   = icon;
            document.getElementById('fieldRiskLabel').textContent  = label;
            document.getElementById('fieldRiskAction').textContent = action;
            document.getElementById('fieldProbText').textContent   =
                `AI probability: ${prob}%  |  Type: ${data.screening_type || ''}`;

            // Report download button
            const rBtn = document.getElementById('fieldReportBtn');
            if (patientId) {
                rBtn.href = `/report/${patientId}`;
                rBtn.style.display = 'flex';
            } else {
                rBtn.style.display = 'none';
            }

            document.getElementById('fieldResults').style.display = 'block';
        }

        // ── Patient history toggle ──────────────────────────────────
        let _historyLoaded = false;

        function toggleHistory() {
            const wrap = document.getElementById('historyTableWrap');
            const open = wrap.style.display === 'block';
            wrap.style.display = open ? 'none' : 'block';
            if (!open && !_historyLoaded) loadPatientHistory();
        }

        async function loadPatientHistory() {
            try {
                const resp = await fetch('/patients?limit=50');
                const data = await resp.json();
                const rows = data.patients || [];
                const tbody = document.getElementById('historyTableBody');
                if (rows.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--gray-500);padding:12px;">No screenings recorded yet.</td></tr>';
                } else {
                    tbody.innerHTML = rows.map(r => {
                        const risk = (r.risk_level || 'low').toLowerCase();
                        const riskClass = risk === 'high' ? 'hist-risk-high' : (risk === 'medium' || risk === 'moderate') ? 'hist-risk-medium' : 'hist-risk-low';
                        const date = r.screened_at ? r.screened_at.split('T')[0] : '—';
                        return `<tr>
                            <td>${date}</td>
                            <td>${r.name || '—'}</td>
                            <td>${r.age || '—'}</td>
                            <td>${r.village || '—'}</td>
                            <td>${r.screening_type || '—'}</td>
                            <td class="${riskClass}">${(r.risk_level || '').toUpperCase()}</td>
                            <td><a href="/report/${r.patient_id}" target="_blank" style="color:#2563eb;font-size:0.8rem;">PDF</a></td>
                        </tr>`;
                    }).join('');
                }
                _historyLoaded = true;
            } catch (e) {
                document.getElementById('historyTableBody').innerHTML =
                    '<tr><td colspan="7" style="text-align:center;color:#dc2626;padding:12px;">Could not load history.</td></tr>';
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    if metadata:
        auc_score = f"{metadata['performance']['auc']:.4f}"
        sensitivity = f"{metadata['performance']['sensitivity']*100:.1f}"
        specificity = f"{metadata['performance']['specificity']*100:.1f}"
        optimal_threshold = f"{metadata['performance']['optimal_threshold']:.3f}"
        training_date = metadata.get('training_date', 'Unknown')
    else:
        auc_score = "0.9889"
        sensitivity = "92.96"
        specificity = "97.01"
        optimal_threshold = "0.512"
        training_date = "Unknown"
    
    html = HTML_TEMPLATE.replace('{{ auc_score }}', auc_score)\
                        .replace('{{ sensitivity }}', sensitivity)\
                        .replace('{{ specificity }}', specificity)\
                        .replace('{{ optimal_threshold }}', optimal_threshold)\
                        .replace('{{ training_date }}', training_date)
    
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    if not ORAL_DEPENDENCIES_OK:
        return jsonify({'error': 'Oral screening dependencies not available'}), 500
    
    if oral_model is None:
        success, msg = load_oral_model_and_metadata()
        if not success:
            return jsonify({'error': f'Model load failed: {msg}'}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not _allowed_image(file.filename):
            return jsonify({
                'error': f'Unsupported file type. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}'
            }), 400

        mode = request.form.get('mode', 'diagnostic')
        image_bytes = file.read()
        
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("OpenCV could not decode image")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return jsonify({'error': f'Cannot read image file: {e}'}), 400
        
        img_uint8 = cv2.resize(img, (224, 224))  # kept for Grad-CAM overlay
        img = img_uint8.astype('float32') / 255.0
        img_array = np.expand_dims(img, axis=0)
        
        cancer_prob = float(oral_model.predict(img_array, verbose=0)[0][0])
        non_cancer_prob = 1 - cancer_prob
        
        optimal_threshold = 0.512
        if mode == 'screening':
            threshold = optimal_threshold * 0.7
        else:
            threshold = optimal_threshold
        
        is_cancer = cancer_prob >= threshold
        prediction = "SUSPICIOUS PATTERN" if is_cancer else "NORMAL PATTERN"
        
        confidence = cancer_prob if is_cancer else non_cancer_prob
        recommendation = get_oral_recommendation(cancer_prob, is_cancer, mode)
        
        if prediction == "SUSPICIOUS PATTERN":
            if cancer_prob > 0.7:
                risk_level = "HIGH"
            elif cancer_prob > 0.5:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
        else:
            if cancer_prob < 0.2:
                risk_level = "VERY LOW"
            elif cancer_prob < 0.4:
                risk_level = "LOW"
            else:
                risk_level = "BORDERLINE"
        
        # Grad-CAM explainability for v1 model
        gradcam_b64 = None
        if V2_AVAILABLE:
            try:
                heatmap, overlay = v2_gradcam(
                    model=oral_model,
                    image=img_array,   # (1,224,224,3) in [0,1]
                    metadata=None,     # single-input model
                    layer_name=None,   # auto-detect last conv layer
                    alpha=0.4,
                )
                if overlay is not None:
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    _, buf = cv2.imencode('.png', overlay_bgr)
                    if buf is not None and len(buf) > 0:
                        gradcam_b64 = base64.b64encode(buf).decode('utf-8')
            except Exception as _e:
                print(f'[v1] Grad-CAM warning: {_e}')

        return jsonify({
            'success': True,
            'cancer_probability': cancer_prob,
            'non_cancer_probability': non_cancer_prob,
            'prediction': prediction,
            'confidence': confidence,
            'threshold_used': threshold,
            'mode': mode.upper(),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'gradcam_png_b64': gradcam_b64,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system': 'oral_screening_v1'
        })
        
    except Exception as e:
        print(f"Error in oral prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict/skin', methods=['POST'])
def predict_skin_route():
    if not SKIN_MODULE_AVAILABLE:
        return jsonify({
            'error': 'Skin screening module not available',
            'details': 'Please ensure modules/skin_screening.py exists and all its dependencies are installed'
        }), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if not _allowed_image(file.filename):
        return jsonify({
            'error': f'Unsupported file type. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}'
        }), 400

    try:
        image_bytes = file.read()
        result = predict_skin(image_bytes)
        
        if 'prediction' in result and 'CANCER' in str(result['prediction']).upper():
            result['prediction'] = 'SUSPICIOUS PATTERN'
        elif 'prediction' in result and 'NON' in str(result['prediction']).upper():
            result['prediction'] = 'BENIGN PATTERN'
        
        if 'cancer_probability' in result:
            prob = result['cancer_probability']
            if prob > 0.7:
                result['risk_level'] = 'HIGH'
            elif prob > 0.4:
                result['risk_level'] = 'MODERATE'
            else:
                result['risk_level'] = 'LOW'
        
        result.update({
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system': 'skin_screening',
            'disclaimer': 'This is an AI-assisted screening tool. Results should be confirmed by a qualified healthcare professional. Not for diagnosis.'
        })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in skin prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_v2', methods=['POST'])
def predict_v2():
    """
    Multi-modal prediction endpoint (CuraLens v2/v3).

    Accepts multipart/form-data with:
        image        : image file (JPEG / PNG)
        cancer_type  : "oral_legacy" (default) | "oral" | "skin"

    Metadata fields for cancer_type="oral_legacy" (backward-compat):
        age, smoking (0/1), alcohol (0/1), sun_exposure (0-10)

    Metadata fields for cancer_type="oral":
        age, smoking_years, cigarettes_per_day,
        alcohol_units_per_week, chewing_tobacco (0/1), family_history (0/1)

    Metadata fields for cancer_type="skin":
        age, skin_type (1-6), sunburn_history,
        outdoor_hours_per_week, tanning_bed_use (0/1), family_history (0/1)

    Alternatively, pass a single JSON-encoded `metadata` form field.

    Returns structured JSON with prediction, risk tier, Grad-CAM PNG,
    and any metadata validation warnings.

    NOTE: /predict (v1 oral) and /predict/skin (v1 skin) are untouched.
    """
    if not V2_AVAILABLE:
        return jsonify({
            'error': 'CuraLens v2 modules are not available',
            'hint': 'Ensure models_v2/ and utils_v2/ directories are present'
        }), 503

    # ---------------------------------------------------------------- routing
    cancer_type = request.form.get('cancer_type', 'oral_legacy').strip().lower()

    VALID_CANCER_TYPES = ('oral_legacy', 'oral', 'skin')
    if cancer_type not in VALID_CANCER_TYPES:
        return jsonify({
            'error': f"Invalid cancer_type='{cancer_type}'. "
                     f"Valid options: {list(VALID_CANCER_TYPES)}"
        }), 400

    # ----------------------------------------------------------------- model
    if cancer_type == 'skin':
        ok, msg = _load_skin_v3_model()
        active_model = _skin_v3_model
        system_tag   = 'skin_screening_v3'
    elif cancer_type == 'oral':
        ok, msg = _load_oral_v3_model()
        active_model = _oral_v3_model
        system_tag   = 'oral_screening_v3'
    else:
        # oral_legacy — backward compatible default
        ok, msg = _load_v2_model()
        active_model = _v2_model
        system_tag   = 'oral_screening_v2'

    if not ok or active_model is None:
        # Skin V2 not yet trained → graceful V1 fallback while training runs
        if cancer_type == 'skin' and SKIN_MODULE_AVAILABLE:
            if 'image' not in request.files or request.files['image'].filename == '':
                return jsonify({'error': 'No image uploaded'}), 400
            try:
                fb_bytes = request.files['image'].read()
                v1_result = predict_skin(fb_bytes)
            except Exception as e:
                return jsonify({'error': f'Skin V1 fallback error: {e}'}), 500
            raw_prob = float(np.clip(float(v1_result.get('cancer_probability', 0.5)), 0.0, 1.0))
            risk = v2_score(raw_prob)
            return jsonify({
                'success'          : True,
                'version'          : 'v1_fallback_skin',
                'cancer_type'      : 'skin',
                'probability'      : round(raw_prob, 6),
                'probability_pct'  : round(raw_prob * 100, 2),
                'risk_level'       : risk.risk_level,
                'risk_label'       : risk.risk_label,
                'confidence_band'  : risk.confidence_band,
                'recommendation'   : risk.recommendation,
                'color_code'       : risk.color_code,
                'metadata_used'    : {},
                'metadata_warnings': [
                    'Skin V2 multimodal model is still training — using V1 image-only model as fallback. '
                    'Metadata risk factors are not included in this result.'
                ],
                'gradcam_png_b64'  : None,
                'gradcam_shape'    : None,
                'timestamp'        : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'system'           : 'skin_screening_v1_fallback',
                'disclaimer'       : (
                    'AI-assisted screening tool. '
                    'V2 multimodal model is currently training — result uses V1 image-only model. '
                    'Not validated for clinical use. Consult a qualified healthcare professional.'
                ),
            })
        return jsonify({'error': f'Model load failed [{cancer_type}]: {msg}'}), 500

    # ------------------------------------------------------------------ image
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded. Send the file under the key "image".'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError('OpenCV could not decode the uploaded image')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({'error': f'Image decode error: {e}'}), 400

    img_rgb_resized = cv2.resize(img_rgb, (224, 224))
    # NOTE: Do NOT divide by 255 here.  EfficientNetB0 (used in the v2 model)
    # includes its own internal Rescaling + Normalization preprocessing layers
    # and expects raw pixel values in [0, 255].  Dividing by 255 first causes
    # double-rescaling that collapses all images to near-zero, making the model
    # output the same value (~0) for every image regardless of content.
    img_batch = np.expand_dims(img_rgb_resized.astype('float32'), axis=0)  # (1,224,224,3)

    # --------------------------------------------------------------- metadata
    meta_warnings: list = []
    try:
        raw_meta = request.form.get('metadata', None)
        if raw_meta:
            meta_dict = json.loads(raw_meta)
        else:
            # Build dict from individual form fields depending on cancer_type
            if cancer_type == 'skin':
                meta_dict = {
                    'age'                    : request.form.get('age',                    0),
                    'skin_type'              : request.form.get('skin_type',              3),
                    'sunburn_history'        : request.form.get('sunburn_history',        2),
                    'outdoor_hours_per_week' : request.form.get('outdoor_hours_per_week', 10),
                    'tanning_bed_use'        : request.form.get('tanning_bed_use',        0),
                    'family_history'         : request.form.get('family_history',         0),
                }
            elif cancer_type == 'oral':
                meta_dict = {
                    'age'                    : request.form.get('age',                    0),
                    'smoking_years'          : request.form.get('smoking_years',          0),
                    'cigarettes_per_day'     : request.form.get('cigarettes_per_day',     0),
                    'alcohol_units_per_week' : request.form.get('alcohol_units_per_week', 0),
                    'chewing_tobacco'        : request.form.get('chewing_tobacco',        0),
                    'family_history'         : request.form.get('family_history',         0),
                }
            else:
                # oral_legacy — 4-field backward compat
                meta_dict = {
                    'age'          : request.form.get('age',          0),
                    'smoking'      : request.form.get('smoking',      0),
                    'alcohol'      : request.form.get('alcohol',      0),
                    'sun_exposure' : request.form.get('sun_exposure', 0),
                }

        # Validate and encode using clinical schema
        metadata_array, meta_warnings = _validate_meta(
            meta_dict, cancer_type=cancer_type, normalize=True
        )

    except Exception as e:
        return jsonify({'error': f'Metadata parse/validation error: {e}'}), 400

    # Determine the correct input names based on model
    try:
        input_names = [inp.name.split(':')[0] for inp in active_model.inputs]
        # Build appropriately-named dict for multi-input models
        if len(input_names) == 2:
            img_key  = input_names[0]   # e.g. "oral_image_input" or "image_input"
            meta_key = input_names[1]
            model_input = [img_batch, metadata_array]
        else:
            model_input = img_batch
    except Exception:
        model_input = [img_batch, metadata_array]

    # ------------------------------------------------------------ prediction
    try:
        raw_prob = float(active_model.predict(model_input, verbose=0)[0][0])
        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))

        # Guard: if the model outputs essentially 0 for everything (collapsed model),
        # fall back to the v1 oral model so the user gets a calibrated result.
        if raw_prob < 1e-10 and cancer_type in ('oral_legacy', 'oral') and ORAL_DEPENDENCIES_OK:
            print('[v2] WARNING: v2 model output is near-zero (model may need retraining). '
                  'Falling back to v1 oral model for image probability.')
            v1_batch = img_batch / 255.0   # v1 model expects [0,1]
            raw_prob = float(oral_model.predict(v1_batch, verbose=0)[0][0])
            raw_prob = float(np.clip(raw_prob, 0.0, 1.0))
            system_tag += '_v1_fallback'
            meta_warnings.insert(0,
                'v2 multimodal model is currently being recalibrated. '
                'Image probability from v1 model; metadata risk factors applied separately.')
    except Exception as e:
        print(f'[v2] Prediction error: {e}')
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    # ---------------------------------------------------------- risk scoring
    risk = v2_score(raw_prob)

    # --------------------------------------------------------------- grad-cam
    gradcam_b64  = None
    gradcam_shape = None
    try:
        heatmap, overlay = v2_gradcam(
            model      = active_model,
            image      = img_batch,
            metadata   = metadata_array,
            layer_name = None,    # auto-detect last conv layer
            save_path  = None,
            alpha      = 0.4,
        )
        if overlay is not None:
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, buf       = cv2.imencode('.png', overlay_bgr)
            if buf is not None and len(buf) > 0:
                gradcam_b64   = base64.b64encode(buf).decode('utf-8')
                gradcam_shape = list(heatmap.shape) if heatmap is not None else None
        else:
            print('[v2] Grad-CAM returned None overlay (blank heatmap guard).')
    except Exception as e:
        print(f'[v2] Grad-CAM warning: {e}')
        gradcam_b64 = None

    # -------------------------------------------------------------- response
    return jsonify({
        'success'          : True,
        'version'          : f'v2_multimodal_{cancer_type}',
        'cancer_type'      : cancer_type,
        # ── core prediction ──
        'probability'      : round(raw_prob, 6),
        'probability_pct'  : round(raw_prob * 100, 2),
        # ── risk tier ──
        'risk_level'       : risk.risk_level,
        'risk_label'       : risk.risk_label,
        'confidence_band'  : risk.confidence_band,
        'recommendation'   : risk.recommendation,
        'color_code'       : risk.color_code,
        # ── inputs echoed back ──
        'metadata_used'    : meta_dict,
        'metadata_warnings': meta_warnings,
        # ── explainability ──
        'gradcam_png_b64'  : gradcam_b64,
        'gradcam_shape'    : gradcam_shape,
        # ── meta ──
        'timestamp'        : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system'           : system_tag,
        'disclaimer'       : (
            'AI-assisted screening tool. '
            'Not validated for clinical use. '
            'Always consult a qualified healthcare professional.'
        ),
    })


@app.route('/schema/<cancer_type>', methods=['GET'])
def get_metadata_schema(cancer_type: str):
    """
    Return the metadata field schema for a given cancer type.
    Useful for dynamic frontend form generation.

    GET /schema/oral_legacy
    GET /schema/oral
    GET /schema/skin
    """
    if not V2_AVAILABLE:
        return jsonify({'error': 'v2 modules not available'}), 503
    try:
        schema = _get_schema_info(cancer_type)
        return jsonify({'cancer_type': cancer_type, 'fields': schema})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/screen', methods=['POST'])
def field_screen():
    """
    Field Worker Screening endpoint.

    Accepts multipart/form-data:
        image            : image file (JPEG / PNG)
        screening_type   : "oral" (default) | "skin"
        name             : patient name
        age              : patient age
        gender           : Male / Female / Other
        village          : village or area
        contact          : phone number (optional)
        screened_by      : health worker name (optional)

    Returns JSON with full prediction result + patient_id for later PDF download.
    """
    if not ORAL_DEPENDENCIES_OK:
        return jsonify({'error': 'Screening dependencies not available'}), 500

    if oral_model is None:
        ok, msg = load_oral_model_and_metadata()
        if not ok:
            return jsonify({'error': f'Model load failed: {msg}'}), 500

    # ── validate image ────────────────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    if not _allowed_image(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}'}), 400

    screening_type = request.form.get('screening_type', 'oral').strip().lower()

    # ── patient info ──────────────────────────────────────────────────────────
    patient_info = {
        'name':        request.form.get('name', 'Anonymous').strip(),
        'age':         request.form.get('age', '').strip(),
        'gender':      request.form.get('gender', '').strip(),
        'village':     request.form.get('village', '').strip(),
        'contact':     request.form.get('contact', '').strip(),
        'screened_by': request.form.get('screened_by', '').strip(),
    }

    try:
        image_bytes = file.read()
        nparr  = np.frombuffer(image_bytes, np.uint8)
        img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('Could not decode uploaded image')
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({'error': f'Image decode error: {e}'}), 400

    img_uint8  = cv2.resize(img, (224, 224))
    img_norm   = img_uint8.astype('float32') / 255.0
    img_batch  = np.expand_dims(img_norm, axis=0)

    # ── choose model based on screening_type ─────────────────────────────────
    if screening_type == 'skin' and SKIN_MODULE_AVAILABLE:
        result_raw = predict_skin(image_bytes)
        cancer_prob = float(result_raw.get('cancer_probability',
                            result_raw.get('risk_score', 0.5)))
        cancer_prob = float(np.clip(cancer_prob, 0.0, 1.0))
    else:
        cancer_prob = float(oral_model.predict(img_batch, verbose=0)[0][0])
        cancer_prob = float(np.clip(cancer_prob, 0.0, 1.0))

    # ── risk tier ─────────────────────────────────────────────────────────────
    if cancer_prob >= 0.7:
        risk_level = 'HIGH';   risk_label = 'High Risk — Refer Immediately'
        recommendation = ('URGENT: High-risk pattern detected. '
                          'Refer patient to a qualified doctor or specialist immediately. '
                          'Do not delay evaluation.')
        color_code = '#ef4444'
    elif cancer_prob >= 0.3:
        risk_level = 'MEDIUM'; risk_label = 'Medium Risk — Further Evaluation Needed'
        recommendation = ('CAUTION: Abnormal pattern detected. '
                          'Schedule a follow-up with a doctor within 2–4 weeks. '
                          'Monitor closely.')
        color_code = '#f59e0b'
    else:
        risk_level = 'LOW';    risk_label = 'Low Risk — Routine Monitoring'
        recommendation = ('NORMAL: No high-risk patterns detected. '
                          'Continue routine annual health screenings. '
                          'Maintain good oral hygiene / skin care.')
        color_code = '#22c55e'

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam_b64   = None
    gradcam_bytes_store = None
    if V2_AVAILABLE:
        try:
            heatmap, overlay = v2_gradcam(
                model=oral_model, image=img_batch,
                metadata=None, layer_name=None, alpha=0.4,
            )
            if overlay is not None:
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                _, buf      = cv2.imencode('.png', overlay_bgr)
                if buf is not None and len(buf) > 0:
                    gradcam_bytes_store = buf.tobytes()
                    gradcam_b64 = base64.b64encode(gradcam_bytes_store).decode('utf-8')
        except Exception as _e:
            print(f'[screen] Grad-CAM warning: {_e}')

    # ── save patient record ───────────────────────────────────────────────────
    record = {
        'patient_info':    patient_info,
        'screening_type':  screening_type,
        'screening_date':  datetime.now().strftime('%Y-%m-%d %H:%M'),
        'probability':     round(cancer_prob, 6),
        'probability_pct': round(cancer_prob * 100, 2),
        'risk_level':      risk_level,
        'risk_label':      risk_label,
        'recommendation':  recommendation,
        'color_code':      color_code,
        'model_used':      f'oral_v1_{"skin_v1" if screening_type == "skin" else "oral_v1"}',
        'metadata_used':   {},
        'disclaimer':      ('AI-assisted screening only. Not a medical diagnosis. '
                            'Always consult a qualified healthcare professional.'),
    }
    patient_id = _save_patient_record(record)

    # ── store images on disk for PDF generation ───────────────────────────────
    img_dir = os.path.join(PATIENTS_DIR, 'images')
    os.makedirs(img_dir, exist_ok=True)
    try:
        _, orig_buf = cv2.imencode('.jpg', cv2.resize(
            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), (224, 224)))
        with open(os.path.join(img_dir, f'{patient_id}_orig.jpg'), 'wb') as fh:
            fh.write(orig_buf.tobytes())
    except Exception: pass
    if gradcam_bytes_store:
        try:
            with open(os.path.join(img_dir, f'{patient_id}_gcam.png'), 'wb') as fh:
                fh.write(gradcam_bytes_store)
        except Exception: pass

    record['patient_id'] = patient_id
    return jsonify({
        'success':         True,
        'patient_id':      patient_id,
        'patient_name':    patient_info['name'],
        'screening_type':  screening_type,
        'probability':     round(cancer_prob, 6),
        'probability_pct': round(cancer_prob * 100, 2),
        'risk_level':      risk_level,
        'risk_label':      risk_label,
        'recommendation':  recommendation,
        'color_code':      color_code,
        'gradcam_png_b64': gradcam_b64,
        'report_url':      f'/report/{patient_id}',
        'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'disclaimer':      record['disclaimer'],
    })


@app.route('/report/<patient_id>', methods=['GET'])
def download_report(patient_id: str):
    """
    Generate and stream a printable PDF screening report.
    GET /report/<patient_id>
    """
    # Sanitise ID — only uppercase alphanum + hyphen allowed
    safe_id = ''.join(c for c in patient_id.upper() if c.isalnum() or c == '-')
    record = _load_patient_record(safe_id)
    if record is None:
        return jsonify({'error': f'Patient record not found: {safe_id}'}), 404

    img_dir = os.path.join(PATIENTS_DIR, 'images')
    orig_path = os.path.join(img_dir, f'{safe_id}_orig.jpg')
    gcam_path = os.path.join(img_dir, f'{safe_id}_gcam.png')

    image_bytes   = open(orig_path, 'rb').read() if os.path.exists(orig_path) else None
    gradcam_bytes = open(gcam_path, 'rb').read()  if os.path.exists(gcam_path) else None

    try:
        pdf_buf = _build_pdf_report(record, image_bytes, gradcam_bytes)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'PDF generation failed: {e}'}), 500

    fname = f"CuraLens_Report_{safe_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return send_file(
        pdf_buf,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=fname,
    )


@app.route('/patients', methods=['GET'])
def list_patients():
    """
    Return patient records as JSON (most recent first).
    GET /patients?limit=50
    Returns list of patient records (no image data, metadata only).
    """
    try:
        limit = min(int(request.args.get('limit', 100)), 500)
    except ValueError:
        limit = 100
    records = _list_patient_records(limit=limit)
    return jsonify({'count': len(records), 'patients': records})


@app.route('/health', methods=['GET'])
def health():
    """
    System health endpoint — returns the load status of every model and module.

    GET /health
    Returns JSON with:
        status           : "ok" | "degraded"
        models           : dict of model availability flags
        uptime_timestamp : server start ISO timestamp
    """
    models_status = {
        'oral_v1': {
            'loaded': oral_model is not None,
            'source': 'models/oral_cancer_model.h5',
            'auc': metadata['performance']['auc'] if metadata else None,
        },
        'skin_v1_module': {
            'available': SKIN_MODULE_AVAILABLE,
            'source': 'modules/skin_screening.py',
        },
        'oral_v2_multimodal': {
            'loaded': _v2_model is not None,
            'source': 'models_v2/saved_model/',
        },
        'oral_v3_multimodal': {
            'loaded': _oral_v3_model is not None,
            'source': 'models_v2/oral_saved_model/',
        },
        'skin_v3_multimodal': {
            'loaded': _skin_v3_model is not None,
            'source': 'models_v2/skin_saved_model/',
        },
    }

    v2_modules_ok = V2_AVAILABLE
    oral_ok = ORAL_DEPENDENCIES_OK and oral_model is not None
    overall = 'ok' if (oral_ok and v2_modules_ok) else 'degraded'

    return jsonify({
        'status': overall,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modules': {
            'v2_available': V2_AVAILABLE,
            'oral_dependencies_ok': ORAL_DEPENDENCIES_OK,
            'skin_module_available': SKIN_MODULE_AVAILABLE,
        },
        'models': models_status,
        'endpoints': ['/predict', '/predict/skin', '/predict_v2', '/schema/<cancer_type>', '/health'],
    })


if __name__ == '__main__':
    import sys

    port = 5001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except:
            pass
    
    print("="*60)
    print("🚀 CuraLens – AI-Assisted Multi-Module Screening Platform")
    print("="*60)
    print("🦷 Oral Screening v1 : Active")
    print("🧴 Skin Screening    : " + ("Active" if SKIN_MODULE_AVAILABLE else "Not Available"))
    print("🔬 Multimodal v2     : " + ("Available" if V2_AVAILABLE else "Not Available"))
    print("="*60)
    
    if not ORAL_DEPENDENCIES_OK:
        print("❌ Missing dependencies for oral screening. Please install:")
        print("   pip install tensorflow opencv-python pillow flask")
        print("   Oral screening will be unavailable.")
    
    if SKIN_MODULE_AVAILABLE:
        print("✅ Skin screening module loaded")
    else:
        print("⚠️  Skin screening module not available")
    
    if ORAL_DEPENDENCIES_OK:
        success, msg = load_oral_model_and_metadata()
        if success:
            print(f"✅ {msg}")
            print(f"📊 Model AUC: {metadata['performance']['auc']:.4f}" if metadata else "📊 Using default model info")
        else:
            print(f"❌ Failed to load oral model: {msg}")
            print("⚠️  Oral screening will be unavailable.")
    
    print(f"\n🌐 Starting web server on port {port}...")
    print(f"📡 Open your browser and go to: http://localhost:{port}")
    print("="*60)
    print("\n⚠️  IMPORTANT: This is an AI-assisted SCREENING tool only.")
    print("   Not for diagnosis. Always consult healthcare professionals.")
    print("="*60)

    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=debug_mode)