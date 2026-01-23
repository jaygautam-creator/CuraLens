# CuraLens 🩺  
### AI-Assisted Oral Image Screening System

CuraLens is a deep learning–based web application designed to assist in the **screening of oral cavity images** for **abnormal patterns**.  
It functions as an **AI-assisted decision-support system** and **does not provide medical diagnosis** or replace clinical judgment.

---

## 🔍 Project Motivation
Oral cancer and other oral abnormalities have a high prevalence in countries like India.  
Early-stage **screening and risk flagging** can help guide individuals toward timely clinical evaluation.

This project explores how **computer vision and transfer learning** can support **preliminary screening** using oral cavity images in an ethical and responsible manner.

---

## 🚀 Key Features
- Binary image classification: **Normal vs Abnormal**
- Deep learning model using **MobileNetV2 (transfer learning)**
- Outputs **risk probability score (0–1)** for abnormality
- Interactive **web application** for image-based screening
- **CLI-based prediction tool** for experimentation
- Adjustable **screening vs confirmation thresholds**
- Strong emphasis on **clinical disclaimers and ethical use**

---

## 🧠 Model Overview
- **Architecture:** MobileNetV2 + custom dense classifier  
- **Input size:** 224 × 224 RGB  
- **Output:** Abnormality probability (0–1)  
- **Loss function:** Binary Cross-Entropy  

### Training Strategy
- Class weighting to handle imbalance  
- Data augmentation  
- Early stopping  
- Learning rate scheduling  

---

## 📂 Project Structure

```
OralCancerApp/
├── train.py              # Model training script
├── predict.py            # CLI-based prediction
├── web_app.py            # Flask web application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore
└── models/
    └── model_metadata.json
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run the Web Application
```bash
python web_app.py
```

Open in browser:  
👉 http://localhost:5001

---

### 3️⃣ Run CLI Prediction
```bash
python predict.py path/to/image.jpg
```

Optional custom threshold:
```bash
python predict.py image.jpg 0.35
```

---

## ⚠️ Disclaimer (Important)

This system is developed **strictly for educational and research purposes**.

- The model performs **image-based screening only**
- It does **not diagnose cancer or any disease**
- Results must **always be reviewed by qualified medical professionals**
- Clinical decisions **must not** be made based on this tool alone

---

## 🔮 Future Scope
- Multi-class oral abnormality categorization  
- Grad-CAM visual explanations for model interpretability  
- Secure REST API for research integration  
- Mobile application interface  
- Expansion to other medical imaging domains (research-only)

---

## 👨‍🎓 Author

**Jay Gautam**  
B.Tech – Computer Science (Artificial Intelligence & Machine Learning)

---

## 🟢 Project Status
- Core AI screening system: ✅ Completed  
- UI and usability improvements: ✅ Completed  
- Workflow automation & logging: 🟡 Under experimentation  
- Clinical validation: ❌ Not included (out of scope)

---

> *CuraLens is a technical exploration of AI-assisted screening, designed with responsibility, transparency, and academic integrity at its core.*
