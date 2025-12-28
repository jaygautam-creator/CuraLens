"""
Oral Cancer Detection Web App - CORRECTED VERSION
Run: python web_app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import os
import io
import numpy as np
import json
from datetime import datetime
import traceback

# Try imports
try:
    import tensorflow as tf
    import cv2
    from PIL import Image
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    print(f"❌ Missing dependency: {e}")

app = Flask(__name__)
model = None
metadata = None

def load_model_and_metadata():
    """Load model and metadata."""
    global model, metadata
    
    try:
        # Load model
        model_path = 'models/oral_cancer_model.h5'
        if not os.path.exists(model_path):
            print(f"⚠️ {model_path} not found, trying best_model.h5...")
            model_path = 'models/best_model.h5'
        
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded from {model_path}")
        
        # Load metadata
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
        
        return True, "Model and metadata loaded"
        
    except Exception as e:
        return False, str(e)

def preprocess_image(image_bytes):
    """Preprocess uploaded image."""
    # Convert to numpy array
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:  # RGB
        pass  # Already RGB
    
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    
    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0
    
    return np.expand_dims(image, axis=0)

def get_threshold_for_mode(mode, optimal_threshold=0.512):
    """Get threshold based on mode."""
    if mode == 'screening':
        # Lower threshold = higher sensitivity (detect more cancer)
        return optimal_threshold * 0.7  # More sensitive
    else:  # diagnostic
        return optimal_threshold

def get_recommendation(cancer_prob, is_cancer, mode):
    """Get clinical recommendation."""
    if is_cancer:
        if cancer_prob > 0.7:
            return "HIGH RISK: Strong indication of oral cancer. Immediate biopsy and specialist consultation recommended."
        elif cancer_prob > 0.5:
            return "MODERATE RISK: Possible oral cancer. Biopsy recommended with close monitoring."
        else:
            return "LOW RISK: Slight indication of oral cancer. Clinical examination and follow-up in 2-3 months recommended."
    else:
        if cancer_prob < 0.2:
            return "VERY LOW RISK: Unlikely to be cancer. Routine annual screening recommended."
        elif cancer_prob < 0.4:
            return "LOW RISK: Probably benign. Follow-up in 6-12 months recommended."
        else:
            return "BORDERLINE: No cancer detected but requires monitoring. Follow-up in 3-6 months recommended."

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Oral Cancer AI Detection</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 600px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #2d3748;
            font-size: 32px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .header p {
            color: #718096;
            font-size: 16px;
        }
        
        .upload-area {
            border: 3px dashed #cbd5e0;
            border-radius: 15px;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s;
            cursor: pointer;
            background: #f7fafc;
        }
        
        .upload-area:hover, .upload-area.dragover {
            border-color: #667eea;
            background: #edf2f7;
        }
        
        .upload-icon {
            font-size: 48px;
            color: #a0aec0;
            margin-bottom: 15px;
        }
        
        .upload-text h3 {
            color: #4a5568;
            margin-bottom: 5px;
        }
        
        .upload-text p {
            color: #a0aec0;
            font-size: 14px;
        }
        
        .file-input {
            display: none;
        }
        
        .mode-selection {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .mode-option {
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .mode-option:hover {
            border-color: #667eea;
            background: #f7fafc;
        }
        
        .mode-option.selected {
            border-color: #667eea;
            background: #ebf4ff;
        }
        
        .mode-icon {
            font-size: 24px;
            margin-bottom: 10px;
        }
        
        .mode-title {
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 5px;
        }
        
        .mode-desc {
            font-size: 12px;
            color: #718096;
        }
        
        .analyze-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 18px;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
            margin-bottom: 30px;
        }
        
        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        .analyze-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result {
            background: #f7fafc;
            border-radius: 15px;
            padding: 30px;
            display: none;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .result-title {
            font-size: 24px;
            color: #2d3748;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .probability-meter {
            background: #e2e8f0;
            border-radius: 10px;
            height: 10px;
            margin: 25px 0;
            overflow: hidden;
            position: relative;
        }
        
        .probability-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78 0%, #f6e05e 50%, #f56565 100%);
            border-radius: 10px;
            transition: width 1s ease-out;
        }
        
        .probability-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 5px;
            font-size: 12px;
            color: #718096;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 25px 0;
        }
        
        .metric-box {
            background: white;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 12px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .recommendation {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            border-left: 4px solid #667eea;
        }
        
        .recommendation h4 {
            color: #2d3748;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .recommendation p {
            color: #4a5568;
            line-height: 1.6;
        }
        
        .model-info {
            background: #edf2f7;
            border-radius: 10px;
            padding: 15px;
            margin-top: 25px;
            text-align: center;
            font-size: 12px;
            color: #718096;
        }
        
        .disclaimer {
            margin-top: 20px;
            padding: 15px;
            background: #fff5f5;
            border-radius: 10px;
            border-left: 4px solid #f56565;
            font-size: 14px;
            color: #c53030;
        }
        
        @media (max-width: 480px) {
            .container {
                padding: 20px;
            }
            
            .mode-selection {
                grid-template-columns: 1fr;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦷 Oral Cancer AI Detector</h1>
            <p>Upload an oral lesion image for AI-powered analysis</p>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <input type="file" id="fileInput" class="file-input" accept="image/*">
            <div class="upload-icon">📷</div>
            <div class="upload-text">
                <h3>Click to upload or drag and drop</h3>
                <p>JPG, PNG, or JPEG format (max 5MB)</p>
                <p id="fileName" style="margin-top: 10px; color: #4a5568;"></p>
            </div>
        </div>
        
        <div class="mode-selection">
            <div class="mode-option selected" data-mode="screening" onclick="selectMode('screening')">
                <div class="mode-icon">🔍</div>
                <div class="mode-title">Screening Mode</div>
                <div class="mode-desc">High sensitivity for initial screening</div>
            </div>
            <div class="mode-option" data-mode="diagnostic" onclick="selectMode('diagnostic')">
                <div class="mode-icon">🩺</div>
                <div class="mode-title">Diagnostic Mode</div>
                <div class="mode-desc">Balanced for confirmation</div>
            </div>
        </div>
        
        <button class="analyze-btn" onclick="analyzeImage()" id="analyzeBtn">
            🔬 Analyze Image
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing image with AI...</p>
        </div>
        
        <div class="result" id="result">
            <div class="result-header">
                <div class="result-title" id="resultIconTitle">
                    <!-- Filled by JavaScript -->
                </div>
                <div class="probability-value" id="probabilityValue">
                    <!-- Filled by JavaScript -->
                </div>
            </div>
            
            <div class="probability-meter">
                <div class="probability-fill" id="probabilityFill"></div>
            </div>
            <div class="probability-labels">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-value" id="cancerProb">--%</div>
                    <div class="metric-label">Cancer Probability</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" id="prediction">--</div>
                    <div class="metric-label">Prediction</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" id="confidence">--%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" id="modeUsed">--</div>
                    <div class="metric-label">Mode</div>
                </div>
            </div>
            
            <div class="recommendation">
                <h4>💡 Clinical Recommendation</h4>
                <p id="recommendationText">Upload an image to get recommendation</p>
            </div>
            
            <div class="model-info">
                <p>Model AUC: {{ auc_score }} | Sensitivity: {{ sensitivity }}% | Specificity: {{ specificity }}%</p>
                <p>Optimal threshold: {{ optimal_threshold }} | Last trained: {{ training_date }}</p>
            </div>
            
            <div class="disclaimer">
                <p><strong>⚠️ Important:</strong> This is an AI-assisted tool. All results should be confirmed by a qualified healthcare professional. Do not use for self-diagnosis.</p>
            </div>
        </div>
    </div>
    
    <script>
        let selectedMode = 'screening';
        let selectedFile = null;
        
        // Select mode
        function selectMode(mode) {
            selectedMode = mode;
            document.querySelectorAll('.mode-option').forEach(el => {
                el.classList.remove('selected');
            });
            document.querySelector(`[data-mode="${mode}"]`).classList.add('selected');
        }
        
        // Upload area click
        document.getElementById('uploadArea').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });
        
        // Drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            uploadArea.classList.add('dragover');
        }
        
        function unhighlight() {
            uploadArea.classList.remove('dragover');
        }
        
        // Handle file drop
        uploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }
        
        // Handle file selection
        document.getElementById('fileInput').addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
        
        function handleFiles(files) {
            if (files.length > 0) {
                selectedFile = files[0];
                document.getElementById('fileName').textContent = `Selected: ${selectedFile.name}`;
                uploadArea.style.borderColor = '#48bb78';
            }
        }
        
        // Analyze image
        async function analyzeImage() {
            if (!selectedFile) {
                alert('Please select an image first');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('result').style.display = 'none';
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('mode', selectedMode);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Update UI with results
                updateResults(data);
                
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
        
        function updateResults(data) {
            const cancerProb = (data.cancer_probability * 100).toFixed(1);
            const nonCancerProb = (data.non_cancer_probability * 100).toFixed(1);
            
            // Update probability meter
            document.getElementById('probabilityFill').style.width = cancerProb + '%';
            
            // Update values
            document.getElementById('cancerProb').textContent = cancerProb + '%';
            document.getElementById('prediction').textContent = data.prediction;
            document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('modeUsed').textContent = data.mode;
            document.getElementById('recommendationText').textContent = data.recommendation;
            
            // Update result header
            const resultIconTitle = document.getElementById('resultIconTitle');
            const probabilityValue = document.getElementById('probabilityValue');
            
            let icon, title, color;
            if (data.prediction === 'CANCER') {
                if (data.cancer_probability > 0.7) {
                    icon = '🔴';
                    title = 'HIGH PROBABILITY';
                    color = '#f56565';
                } else if (data.cancer_probability > 0.4) {
                    icon = '🟠';
                    title = 'MODERATE PROBABILITY';
                    color = '#ed8936';
                } else {
                    icon = '🟡';
                    title = 'LOW PROBABILITY';
                    color = '#ecc94b';
                }
            } else {
                if (data.cancer_probability < 0.2) {
                    icon = '🟢';
                    title = 'VERY LOW PROBABILITY';
                    color = '#48bb78';
                } else {
                    icon = '🟡';
                    title = 'LOW PROBABILITY';
                    color = '#ecc94b';
                }
            }
            
            resultIconTitle.innerHTML = `${icon} ${title}`;
            probabilityValue.innerHTML = `<strong style="color: ${color}; font-size: 28px;">${cancerProb}%</strong>`;
            probabilityValue.style.color = color;
            
            // Show result
            document.getElementById('result').style.display = 'block';
            
            // Scroll to result
            document.getElementById('result').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest' 
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Render the home page with model info."""
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
    """Handle image prediction."""
    if not DEPENDENCIES_OK:
        return jsonify({'error': 'Server dependencies not available'}), 500
    
    if model is None:
        success, msg = load_model_and_metadata()
        if not success:
            return jsonify({'error': f'Model load failed: {msg}'}), 500
    
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get mode
        mode = request.form.get('mode', 'diagnostic')
        
        # Preprocess image
        image_bytes = file.read()
        
        # Try to read the image
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("OpenCV could not decode image")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            return jsonify({'error': f'Cannot read image file: {e}'}), 400
        
        # Resize and normalize
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img_array = np.expand_dims(img, axis=0)
        
        # **CRITICAL FIX**: Model outputs CANCER PROBABILITY directly
        # Test shows: Cancer image → 0.6706, which should be cancer probability
        cancer_prob = float(model.predict(img_array, verbose=0)[0][0])
        non_cancer_prob = 1 - cancer_prob
        
        # Get thresholds - use optimal threshold from training (0.512)
        optimal_threshold = 0.512
        if mode == 'screening':
            threshold = optimal_threshold * 0.7  # More sensitive (lower threshold)
        else:  # diagnostic
            threshold = optimal_threshold
        
        # Make prediction: If cancer_prob >= threshold → CANCER, else NON-CANCER
        is_cancer = cancer_prob >= threshold
        prediction = "CANCER" if is_cancer else "NON-CANCER"
        
        # Calculate confidence
        confidence = cancer_prob if is_cancer else non_cancer_prob
        
        # Get recommendation
        recommendation = get_recommendation(cancer_prob, is_cancer, mode)
        
        # Determine risk level
        if prediction == "CANCER":
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
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
# Add this at the end of web_app.py
if __name__ == '__main__':
    import sys
    
    # Get port from command line or use 5001 as default
    port = 5001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except:
            pass
    
    print("="*60)
    print("🦷 Oral Cancer Detection Web App")
    print("="*60)
    
    if not DEPENDENCIES_OK:
        print("❌ Missing dependencies. Please install:")
        print("   pip install tensorflow opencv-python pillow flask")
        exit(1)
    
    # Load model and metadata
    success, msg = load_model_and_metadata()
    if success:
        print(f"✅ {msg}")
        print(f"📊 Model AUC: {metadata['performance']['auc']:.4f}" if metadata else "📊 Using default model info")
        print(f"\n🌐 Starting web server on port {port}...")
        print(f"📡 Open your browser and go to: http://localhost:{port}")
        print("="*60)
        print("\n⚠️  Disclaimer: This tool is for research/educational purposes.")
        print("   Always consult healthcare professionals for medical decisions.")
        print("="*60)
    else:
        print(f"❌ Failed to start: {msg}")
        exit(1)
    
    app.run(host='0.0.0.0', port=port, debug=True)