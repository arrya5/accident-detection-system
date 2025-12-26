"""
Accident Detection System - Training Module

This module provides functionality to train the accident detection model
using transfer learning with MobileNetV2.

Author: [Your Name]
Date: December 2025
License: MIT

Training Strategy:
    Phase 1: Train classification head only (frozen base)
    Phase 2: Fine-tune top 50 layers with cosine decay LR
    Phase 3: Polish entire network with very low LR

Key Techniques:
    - Transfer Learning (MobileNetV2 pre-trained on ImageNet)
    - Label Smoothing (0.1)
    - Progressive Dropout (0.5 -> 0.2)
    - L2 Regularization
    - Data Augmentation (6 types)
    - Cosine Decay Learning Rate Schedule
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import argparse
import os
import sys
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Enable Mixed Precision Training for faster training on modern GPUs
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    MIXED_PRECISION = True
    print("✅ Mixed Precision Training enabled (FP16)")
except:
    MIXED_PRECISION = False
    print("ℹ️ Mixed Precision not available, using FP32")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
IMG_SIZE = 224
BATCH_SIZE = 32  # Increased for larger dataset (was 16)

# Training configuration (optimized for 13K+ images)
PHASE1_EPOCHS = 15  # Reduced from 20 (more data = fewer epochs needed)
PHASE2_EPOCHS = 15  # Reduced from 20
PHASE3_EPOCHS = 5
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-4
LEARNING_RATE_PHASE3 = 1e-6
WARMUP_EPOCHS = 3  # Learning rate warmup epochs

# Regularization
LABEL_SMOOTHING = 0.1
L2_REGULARIZATION = 0.01
DROPOUT_RATES = [0.5, 0.4, 0.3, 0.2]

# Logging
LOG_DIR = "logs"
METRICS_FILE = "training_metrics.json"

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def create_data_augmentation():
    """
    Create data augmentation pipeline.
    
    Augmentations applied:
        - Random horizontal flip
        - Random rotation (±20%)
        - Random zoom (±20%)
        - Random translation (±15%)
        - Random contrast (±20%)
        - Random brightness (±15%)
    
    Returns:
        Keras Sequential model for augmentation
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.15),
    ], name="data_augmentation")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(data_path: str, batch_size: int = BATCH_SIZE):
    """
    Load and prepare the dataset.
    
    Expected directory structure:
        data_path/
        ├── train/
        │   ├── Accident/
        │   └── Non Accident/
        ├── val/
        │   ├── Accident/
        │   └── Non Accident/
        └── test/
            ├── Accident/
            └── Non Accident/
    
    Args:
        data_path: Root path to dataset
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    test_dir = os.path.join(data_path, "test")
    
    # Validate directories exist
    for dir_path, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{name} directory not found: {dir_path}")
    
    print(f"\n📂 Loading dataset from: {data_path}")
    
    # Load datasets
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True,
        seed=42
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    # Print dataset info
    class_names = train_ds.class_names
    print(f"   Classes: {class_names}")
    
    # Count samples per class for class weights
    train_accident = len(list((train_dir / "Accident").iterdir())) if hasattr(train_dir, 'iterdir') else len(os.listdir(os.path.join(train_dir, "Accident")))
    train_non_accident = len(list((train_dir / "Non Accident").iterdir())) if hasattr(train_dir, 'iterdir') else len(os.listdir(os.path.join(train_dir, "Non Accident")))
    total_train = train_accident + train_non_accident
    
    print(f"   Train samples: {total_train} (Accident: {train_accident}, Non-Accident: {train_non_accident})")
    
    # Calculate class weights for balanced training
    class_weights = {
        0: total_train / (2 * train_accident),
        1: total_train / (2 * train_non_accident)
    }
    print(f"   Class weights: {class_weights}")
    
    # Optimize performance with prefetching and caching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_weights, class_names

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_model(data_augmentation: keras.Sequential) -> keras.Model:
    """
    Build the accident detection model.
    
    Architecture:
        Input (224x224x3)
        → Data Augmentation
        → MobileNetV2 (pre-trained, frozen initially)
        → Global Average Pooling
        → BatchNorm → Dropout(0.5)
        → Dense(512) → BatchNorm → ReLU → Dropout(0.4)
        → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
        → Dense(128) → BatchNorm → ReLU → Dropout(0.2)
        → Dense(1) → Sigmoid
    
    Args:
        data_augmentation: Data augmentation layer
        
    Returns:
        Compiled Keras model
    """
    print("\n🏗️ Building model...")
    
    # Load MobileNetV2 with ImageNet weights
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Data augmentation (only during training)
    x = data_augmentation(inputs)
    
    # MobileNetV2 preprocessing
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATES[0])(x)
    
    x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(DROPOUT_RATES[1])(x)
    
    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(DROPOUT_RATES[2])(x)
    
    x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(DROPOUT_RATES[3])(x)
    
    # Final layer - use float32 for numerical stability with mixed precision
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs)
    
    print(f"   Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model, base_model

# ============================================================================
# TRAINING
# ============================================================================

class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup followed by cosine decay.
    
    This helps stabilize training in the early epochs.
    """
    def __init__(self, initial_lr, warmup_steps, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Linear warmup
        warmup_lr = self.initial_lr * (step / warmup_steps)
        
        # Cosine decay after warmup
        decay_steps = total_steps - warmup_steps
        decay_step = step - warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_step / decay_steps))
        decay_lr = self.initial_lr * cosine_decay
        
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }


def train_model(model: keras.Model, base_model: keras.Model,
                train_ds, val_ds, output_path: str, class_weights: dict = None):
    """
    Train the model using three-phase progressive fine-tuning.
    
    Phase 1: Train classification head (base frozen) with warmup
    Phase 2: Fine-tune top 50 layers with cosine decay
    Phase 3: Polish entire network
    
    Args:
        model: Keras model to train
        base_model: MobileNetV2 base model
        train_ds: Training dataset
        val_ds: Validation dataset
        output_path: Path to save trained model
        class_weights: Dictionary of class weights for balanced training
        
    Returns:
        Training history and metrics
    """
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOG_DIR, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,  # Reduced from 10 for faster training
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,  # Reduced from 5
        min_lr=1e-7,
        verbose=1
    )
    
    # Model checkpoint - save best model during training
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=output_path.replace('.keras', '_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # TensorBoard logging
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    histories = []
    all_metrics = {'phases': [], 'timestamp': timestamp}
    
    # Calculate steps for warmup
    steps_per_epoch = len(train_ds)
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    total_steps_phase1 = steps_per_epoch * PHASE1_EPOCHS
    
    # ========================================
    # PHASE 1: Train classification head with warmup
    # ========================================
    print("\n" + "=" * 70)
    print("📚 PHASE 1: Training Classification Head")
    print("   Base model: FROZEN")
    print(f"   Learning rate: {LEARNING_RATE_PHASE1} (with {WARMUP_EPOCHS} epoch warmup)")
    print(f"   Epochs: {PHASE1_EPOCHS}")
    print(f"   Class weights: {class_weights}")
    print("=" * 70)
    
    # Learning rate with warmup
    lr_schedule_phase1 = WarmUpCosineDecay(
        initial_lr=LEARNING_RATE_PHASE1,
        warmup_steps=warmup_steps,
        total_steps=total_steps_phase1
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_phase1),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    history1 = model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop, reduce_lr, checkpoint, tensorboard],
        class_weight=class_weights,
        verbose=1
    )
    histories.append(history1)
    
    # Log Phase 1 metrics
    all_metrics['phases'].append({
        'phase': 1,
        'name': 'Classification Head Training',
        'epochs_completed': len(history1.history['loss']),
        'final_train_acc': float(history1.history['accuracy'][-1]),
        'final_val_acc': float(history1.history['val_accuracy'][-1]),
        'best_val_acc': float(max(history1.history['val_accuracy']))
    })
    
    # ========================================
    # PHASE 2: Fine-tune top layers
    # ========================================
    print("\n" + "=" * 70)
    print("🔧 PHASE 2: Fine-tuning Top Layers")
    print("   Unfreezing last 50 layers of MobileNetV2")
    print(f"   Learning rate: {LEARNING_RATE_PHASE2} with cosine decay")
    print(f"   Epochs: {PHASE2_EPOCHS}")
    print("=" * 70)
    
    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"   Unfroze {trainable_count} layers")
    
    # Cosine decay learning rate
    total_steps_phase2 = steps_per_epoch * PHASE2_EPOCHS
    lr_schedule_phase2 = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE_PHASE2,
        decay_steps=total_steps_phase2,
        alpha=0.01
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_phase2),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_ds,
        epochs=PHASE2_EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop, reduce_lr, checkpoint, tensorboard],
        class_weight=class_weights,
        verbose=1
    )
    histories.append(history2)
    
    # Log Phase 2 metrics
    all_metrics['phases'].append({
        'phase': 2,
        'name': 'Fine-tuning Top Layers',
        'epochs_completed': len(history2.history['loss']),
        'final_train_acc': float(history2.history['accuracy'][-1]),
        'final_val_acc': float(history2.history['val_accuracy'][-1]),
        'best_val_acc': float(max(history2.history['val_accuracy']))
    })
    
    # ========================================
    # PHASE 3: Final polish
    # ========================================
    print("\n" + "=" * 70)
    print("🎯 PHASE 3: Final Polish")
    print("   All layers: TRAINABLE")
    print(f"   Learning rate: {LEARNING_RATE_PHASE3} (very low)")
    print(f"   Epochs: {PHASE3_EPOCHS}")
    print("=" * 70)
    
    # Unfreeze all layers
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = True
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE3),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    history3 = model.fit(
        train_ds,
        epochs=PHASE3_EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint, tensorboard],
        class_weight=class_weights,
        verbose=1
    )
    histories.append(history3)
    
    # Log Phase 3 metrics
    all_metrics['phases'].append({
        'phase': 3,
        'name': 'Final Polish',
        'epochs_completed': len(history3.history['loss']),
        'final_train_acc': float(history3.history['accuracy'][-1]),
        'final_val_acc': float(history3.history['val_accuracy'][-1]),
        'best_val_acc': float(max(history3.history['val_accuracy']))
    })
    
    # Save final model
    model.save(output_path)
    print(f"\n💾 Model saved to: {output_path}")
    
    # Save training metrics to JSON
    metrics_path = os.path.join(log_dir, METRICS_FILE)
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"📊 Training metrics saved to: {metrics_path}")
    
    return histories, all_metrics, log_dir

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model: keras.Model, test_ds, class_names: list, log_dir: str = None) -> dict:
    """
    Evaluate the trained model on test dataset with comprehensive metrics.
    
    Args:
        model: Trained Keras model
        test_ds: Test dataset
        class_names: List of class names
        log_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 70)
    print("📊 EVALUATING ON TEST SET")
    print("=" * 70)
    
    # Basic evaluation
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    
    print(f"\n   Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Generate predictions for detailed metrics
    print("\n   Generating predictions for detailed analysis...")
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_proba.extend(predictions.flatten())
        y_pred.extend((predictions > 0.5).astype(int).flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n   📈 CONFUSION MATRIX:")
    print(f"   {'':<15} Predicted")
    print(f"   {'':<15} {class_names[0]:<12} {class_names[1]:<12}")
    print(f"   Actual {class_names[0]:<6} {cm[0][0]:<12} {cm[0][1]:<12}")
    print(f"   Actual {class_names[1]:<6} {cm[1][0]:<12} {cm[1][1]:<12}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n   📊 DETAILED METRICS:")
    print(f"   Precision (Accident): {precision * 100:.2f}%")
    print(f"   Recall (Accident):    {recall * 100:.2f}%")
    print(f"   F1-Score:             {f1 * 100:.2f}%")
    print(f"   Specificity:          {specificity * 100:.2f}%")
    
    # Classification Report
    print("\n   📋 CLASSIFICATION REPORT:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    for line in report.split('\n'):
        print(f"   {line}")
    
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    # Save evaluation metrics
    if log_dir:
        eval_path = os.path.join(log_dir, 'evaluation_metrics.json')
        with open(eval_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n   💾 Evaluation metrics saved to: {eval_path}")
    
    return metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description='Train Accident Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --data_path /path/to/dataset
  python train.py --data_path /path/to/dataset --epochs 50
  python train.py --data_path /path/to/dataset --output models/custom_model.keras
        """
    )
    
    parser.add_argument('--data_path', '-d', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output', '-o', type=str, 
                        default='models/accident_detector.keras',
                        help='Path to save trained model')
    parser.add_argument('--batch_size', '-b', type=int, default=BATCH_SIZE,
                        help='Training batch size')
    parser.add_argument('--epochs', '-e', type=int, default=None,
                        help='Override epochs for Phase 1')
    
    args = parser.parse_args()
    
    # Override global batch size
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    if args.epochs:
        global PHASE1_EPOCHS
        PHASE1_EPOCHS = args.epochs
    
    print("=" * 70)
    print("🚗 ACCIDENT DETECTION MODEL TRAINING")
    print("   Transfer Learning with MobileNetV2")
    print("=" * 70)
    
    # Load dataset
    try:
        train_ds, val_ds, test_ds, class_weights, class_names = load_dataset(args.data_path, BATCH_SIZE)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Create data augmentation
    data_augmentation = create_data_augmentation()
    
    # Build model
    model, base_model = build_model(data_augmentation)
    model.summary()
    
    # Train model
    histories, training_metrics, log_dir = train_model(
        model, base_model, train_ds, val_ds, args.output, class_weights
    )
    
    # Evaluate model
    eval_metrics = evaluate_model(model, test_ds, class_names, log_dir)
    
    # Save complete metrics
    complete_metrics = {
        'training': training_metrics,
        'evaluation': eval_metrics,
        'config': {
            'batch_size': BATCH_SIZE,
            'phase1_epochs': PHASE1_EPOCHS,
            'phase2_epochs': PHASE2_EPOCHS,
            'phase3_epochs': PHASE3_EPOCHS,
            'warmup_epochs': WARMUP_EPOCHS,
            'label_smoothing': LABEL_SMOOTHING,
            'l2_regularization': L2_REGULARIZATION,
            'dropout_rates': DROPOUT_RATES,
            'mixed_precision': MIXED_PRECISION
        }
    }
    
    final_metrics_path = os.path.join(log_dir, 'complete_training_report.json')
    with open(final_metrics_path, 'w') as f:
        json.dump(complete_metrics, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📈 TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Final Test Accuracy: {eval_metrics['test_accuracy'] * 100:.2f}%")
    print(f"   Precision:           {eval_metrics['precision'] * 100:.2f}%")
    print(f"   Recall:              {eval_metrics['recall'] * 100:.2f}%")
    print(f"   F1-Score:            {eval_metrics['f1_score'] * 100:.2f}%")
    print(f"   Model saved to:      {args.output}")
    print(f"   Logs saved to:       {log_dir}")
    print("=" * 70)
    
    print("\n✅ Training complete! Ready for research paper documentation.")


if __name__ == "__main__":
    main()
