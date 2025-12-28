# predict.py - CORRECTED VERSION
import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import json
from datetime import datetime

def load_model_and_metadata():
    """Load the trained model and metadata"""
    print("🔬 Loading model...")
    
    # Load model
    model = tf.keras.models.load_model('models/oral_cancer_model.h5')
    
    # Load metadata
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Model loaded successfully!")
    print(f"📅 Training date: {metadata['training_date']}")
    print(f"📊 Best AUC: {metadata['performance']['auc']:.4f}")
    
    return model, metadata

def preprocess_image(image_path):
    """Preprocess image for model input"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    
    # Normalize to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_with_correct_labels(model, image_array, metadata):
    """
    Make prediction with correct label interpretation.
    The model outputs probability of class 1 (non_cancer).
    We convert this to probability of cancer.
    """
    # Model outputs probability of NON-CANCER (class 1)
    non_cancer_prob = model.predict(image_array, verbose=0)[0][0]
    
    # Cancer probability = 1 - non_cancer_prob
    cancer_prob = 1 - non_cancer_prob
    
    # Get optimal threshold (this is for non_cancer probability)
    threshold = metadata['performance']['optimal_threshold']
    
    # Decision: if non_cancer_prob >= threshold, predict NON-CANCER, else CANCER
    prediction = "NON-CANCER" if non_cancer_prob >= threshold else "CANCER"
    
    # Confidence is cancer probability if predicting cancer, otherwise non-cancer probability
    confidence = cancer_prob if prediction == "CANCER" else non_cancer_prob
    
    return cancer_prob, non_cancer_prob, prediction, confidence, threshold

def print_results(image_path, cancer_prob, non_cancer_prob, prediction, confidence, threshold):
    """Print formatted results with correct interpretation"""
    print(f"\n" + "="*60)
    print(f"🔍 ANALYSIS: {os.path.basename(image_path)}")
    print("="*60)
    
    print(f"\n📊 PROBABILITY SCORES:")
    print(f"  Probability of CANCER:    {cancer_prob:.4f}")
    print(f"  Probability of NON-CANCER: {non_cancer_prob:.4f}")
    print(f"  Decision threshold:       {threshold:.3f} (for non-cancer)")
    print(f"  Final prediction:         {prediction}")
    print(f"  Confidence:               {confidence:.1%}")
    
    print(f"\n⚖️ DECISION RULE:")
    print(f"  If non-cancer probability ≥ {threshold:.3f} → Predict NON-CANCER")
    print(f"  If non-cancer probability < {threshold:.3f} → Predict CANCER")
    
    print(f"\n🏥 CLINICAL INTERPRETATION:")
    if prediction == "CANCER":
        if cancer_prob > 0.70:
            print(f"  ⚠️ HIGH RISK: Strong indication of oral cancer")
            print(f"  📋 Recommendation: Immediate biopsy and specialist consultation")
        elif cancer_prob > 0.40:
            print(f"  ⚠️ MODERATE RISK: Possible oral cancer")
            print(f"  📋 Recommendation: Biopsy recommended, monitor closely")
        else:
            print(f"  ⚠️ LOW RISK: Slight indication")
            print(f"  📋 Recommendation: Follow-up in 2-3 months")
    else:
        if cancer_prob < 0.10:
            print(f"  ✅ VERY LOW RISK: Unlikely to be cancer")
            print(f"  📋 Recommendation: Routine annual screening")
        elif cancer_prob < 0.30:
            print(f"  ✅ LOW RISK: Probably benign")
            print(f"  📋 Recommendation: Follow-up in 6 months")
        else:
            print(f"  ⚠️ BORDERLINE: No cancer detected but keep monitoring")
            print(f"  📋 Recommendation: Follow-up in 3 months")
    
    # Visual representation
    print(f"\n📈 VISUAL ASSESSMENT:")
    bar_length = 40
    cancer_bar = int(cancer_prob * bar_length)
    print(f"  [{'█'*cancer_bar}{'░'*(bar_length-cancer_bar)}]")
    print(f"  0{' '*(bar_length//2-1)}0.5{' '*(bar_length//2-2)}1.0")
    print(f"  ↑ Probability of cancer (higher = more likely)")
    
    return {
        'image': os.path.basename(image_path),
        'cancer_probability': float(cancer_prob),
        'non_cancer_probability': float(non_cancer_prob),
        'prediction': prediction,
        'confidence': float(confidence),
        'threshold_used': float(threshold),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'Model outputs probability of non-cancer. Cancer probability = 1 - non_cancer_probability.'
    }

def main():
    """Main prediction function"""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("\nExamples:")
        print("  python predict.py image.jpg")
        print("  python predict.py /path/to/image.jpg")
        print("\nThe model outputs probability of NON-CANCER.")
        print("Cancer probability is calculated as 1 - non_cancer_probability.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # Step 1: Load model and metadata
        model, metadata = load_model_and_metadata()
        
        # Step 2: Preprocess image
        img_array = preprocess_image(image_path)
        
        # Step 3: Make prediction (with correct label interpretation)
        cancer_prob, non_cancer_prob, prediction, confidence, threshold = predict_with_correct_labels(
            model, img_array, metadata
        )
        
        # Step 4: Print results
        results = print_results(
            image_path, cancer_prob, non_cancer_prob, prediction, confidence, threshold
        )
        
        # Step 5: Save results
        results_file = f"prediction_{os.path.splitext(os.path.basename(image_path))[0]}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()