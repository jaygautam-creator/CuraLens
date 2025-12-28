# train.py - CORRECTED VERSION
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

print(f"TensorFlow: {tf.__version__}")

def analyze_dataset():
    """Check dataset structure"""
    print("📊 Analyzing dataset...")
    
    train_dir = 'data_clean/train'
    val_dir = 'data_clean/val'
    
    for split_name, split_dir in [('Training', train_dir), ('Validation', val_dir)]:
        print(f"\n{split_name} data:")
        if os.path.exists(split_dir):
            for class_name in ['cancer', 'non_cancer']:
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    count = len([f for f in os.listdir(class_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"  {class_name}: {count} images")
                else:
                    print(f"  ⚠️ {class_name}: directory not found")
        else:
            print(f"  ❌ Directory not found: {split_dir}")
            return False
    return True

def create_data_generators():
    """Create data generators with proper augmentation"""
    print("\n🔄 Creating data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='constant',
        cval=0
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        'data_clean/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_directory(
        'data_clean/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"✅ Classes: {train_gen.class_indices}")
    print(f"📈 Training samples: {train_gen.samples}")
    print(f"📈 Validation samples: {val_gen.samples}")
    
    # Verify cancer is label 1
    cancer_label = None
    for class_name, label in train_gen.class_indices.items():
        if 'cancer' in class_name.lower():
            cancer_label = label
            break
    
    if cancer_label == 1:
        print("✅ Correct: Cancer is label 1 (sigmoid output = cancer probability)")
    else:
        print("⚠️ Warning: Cancer is not label 1")
    
    return train_gen, val_gen

def build_better_model():
    """Build an improved model with transfer learning"""
    print("\n🏗️ Building model...")
    
    # Use MobileNetV2 as base
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build custom classifier
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        base_model,
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision')
        ]
    )
    
    # Print summary
    model.summary()
    
    return model, base_model

def calculate_class_weights(train_gen):
    """Calculate balanced class weights"""
    class_counts = np.bincount(train_gen.classes)
    total = len(train_gen.classes)
    
    # Assuming cancer is class 1
    weight_for_cancer = total / (2.0 * class_counts[1])
    weight_for_non_cancer = total / (2.0 * class_counts[0])
    
    # Give cancer class more importance
    weight_for_cancer *= 2.0
    
    class_weights = {
        0: weight_for_non_cancer,  # non-cancer
        1: weight_for_cancer       # cancer
    }
    
    print(f"\n⚖️ Class weights:")
    print(f"  Non-cancer (class 0): {weight_for_non_cancer:.2f}")
    print(f"  Cancer (class 1): {weight_for_cancer:.2f}")
    
    return class_weights

def train_model(model, train_gen, val_gen, class_weights):
    """Train the model"""
    print("\n🚀 Starting training...")
    
    # Create callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.CSVLogger('models/training_log.csv')
    ]
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    return history

def evaluate_and_save(model, train_gen, val_gen, history):
    """Evaluate model and save results"""
    print("\n📊 Evaluating model...")
    
    # Reset generator
    val_gen.reset()
    
    # Collect predictions
    probs, labels = [], []
    for i in range(len(val_gen)):
        x_batch, y_batch = val_gen[i]
        batch_probs = model.predict(x_batch, verbose=0)
        probs.extend(batch_probs.flatten())
        labels.extend(y_batch)
        
        if (i + 1) * val_gen.batch_size >= val_gen.samples:
            break
    
    probs = np.array(probs)
    labels = np.array(labels)
    
    # Calculate metrics
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    
    # Find optimal threshold (Youden's J)
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx]
    
    # Confusion matrix
    preds = (probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    else:
        accuracy = sensitivity = specificity = precision = f1 = 0
    
    # Print results
    print(f"\n" + "="*50)
    print("📈 FINAL PERFORMANCE")
    print("="*50)
    print(f"AUC Score:          {auc_score:.4f}")
    print(f"Optimal Threshold:  {optimal_threshold:.3f}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Sensitivity:        {sensitivity:.4f}")
    print(f"Specificity:        {specificity:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"F1-Score:           {f1:.4f}")
    
    if cm.shape == (2, 2):
        print(f"\n📊 Confusion Matrix:")
        print(f"True Positives:  {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives:  {tn}")
        print(f"False Negatives: {fn}")
    
    # Save final model
    model.save('models/oral_cancer_model.h5')
    print(f"\n💾 Model saved as: models/oral_cancer_model.h5")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_info': {
            'architecture': 'MobileNetV2 + Custom Classifier',
            'input_shape': [224, 224, 3],
            'output_interpretation': 'sigmoid output = cancer probability'
        },
        'performance': {
            'auc': float(auc_score),
            'optimal_threshold': float(optimal_threshold),
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'confusion_matrix': cm.tolist() if cm.shape == (2, 2) else []
        },
        'dataset': {
            'train_samples': train_gen.samples,
            'val_samples': val_gen.samples,
            'class_mapping': train_gen.class_indices
        }
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("💾 Metadata saved as: models/model_metadata.json")
    
    # Plot training history
    plot_training_history(history)
    
    return {
        'auc': auc_score,
        'threshold': optimal_threshold,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['loss', 'accuracy', 'auc', 'recall']
    titles = ['Loss', 'Accuracy', 'AUC', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(history.history[metric], label=f'Training {title}')
        ax.plot(history.history[f'val_{metric}'], label=f'Validation {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{title} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=100)
    plt.close()
    print("📊 Training history plot saved")

def main():
    """Main training pipeline"""
    print("="*60)
    print("ORAL CANCER DETECTION MODEL TRAINING")
    print("="*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    try:
        # Step 1: Analyze dataset
        if not analyze_dataset():
            return
        
        # Step 2: Create data generators
        train_gen, val_gen = create_data_generators()
        
        # Step 3: Build model
        model, base_model = build_better_model()
        
        # Step 4: Calculate class weights
        class_weights = calculate_class_weights(train_gen)
        
        # Step 5: Train model
        history = train_model(model, train_gen, val_gen, class_weights)
        
        # Step 6: Evaluate and save
        results = evaluate_and_save(model, train_gen, val_gen, history)
        
        print(f"\n" + "="*60)
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        print(f"\n📊 Final AUC: {results['auc']:.4f}")
        print(f"🔬 Sensitivity: {results['sensitivity']:.1%}")
        print(f"🔬 Specificity: {results['specificity']:.1%}")
        print(f"⚙️ Optimal threshold: {results['threshold']:.3f}")
        
        print(f"\n📁 Output files:")
        print(f"  • Model: models/oral_cancer_model.h5")
        print(f"  • Metadata: models/model_metadata.json")
        print(f"  • Training log: models/training_log.csv")
        print(f"  • Training plots: models/training_history.png")
        
        print(f"\n🚀 Next step:")
        print(f"  Test with: python predict.py /path/to/image.jpg")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    main()