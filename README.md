# CuraLens 🩺
## AI-Assisted Oral Cancer Screening System

CuraLens is a deep learning–based web application designed to assist in the screening of oral cancer from oral cavity images.
It is intended as an AI decision-support tool, not a replacement for medical professionals.

🔍 Project Motivation

Oral cancer has a high prevalence in countries like India, where early detection can significantly improve survival rates.
This project explores how computer vision and transfer learning can assist in early screening using medical images.

# 🚀 Features

Binary classification: Cancer vs Non-Cancer

CNN with transfer learning (MobileNetV2)

Command-line prediction tool

Web application for interactive use

Adjustable screening vs diagnostic thresholds

Clear clinical disclaimers

# 🧠 Model Overview

Architecture: MobileNetV2 + custom classifier

Input size: 224 × 224 RGB

Output: Cancer probability (0–1)

Loss: Binary Cross-Entropy

Training strategy:

Class weighting

Data augmentation

Early stopping

Learning rate scheduling

# 📂 Project Structure
OralCancerApp/
├── train.py
├── predict.py
├── web_app.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── model_metadata.json

▶️ How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Web Application
python web_app.py


Open in browser:
http://localhost:5001

3️⃣ Run CLI Prediction
python predict.py path/to/image.jpg


Optional custom threshold:

python predict.py image.jpg 0.35

# ⚠️ Disclaimer

This system is for educational and research purposes only.
Predictions must always be confirmed by qualified medical professionals.

# 🔮 Future Scope

Multi-cancer screening (Breast, Skin, etc.)

Grad-CAM visual explanations

REST API for hospital integration

Mobile application support

# 👨‍🎓 Author

# Jay Gautam
B.Tech – Computer Science (AI & ML)
