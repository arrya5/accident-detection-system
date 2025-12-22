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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
IMG_SIZE = 224
BATCH_SIZE = 16

# Training configuration
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 20
PHASE3_EPOCHS = 5
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE3 = 1e-6

# Regularization
LABEL_SMOOTHING = 0.1
L2_REGULARIZATION = 0.01
DROPOUT_RATES = [0.5, 0.4, 0.3, 0.2]

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
    
    # Optimize performance with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds

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
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    print(f"   Total parameters: {model.count_params():,}")
    
    return model, base_model

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model: keras.Model, base_model: keras.Model,
                train_ds, val_ds, output_path: str):
    """
    Train the model using three-phase progressive fine-tuning.
    
    Phase 1: Train classification head (base frozen)
    Phase 2: Fine-tune top 50 layers with cosine decay
    Phase 3: Polish entire network
    
    Args:
        model: Keras model to train
        base_model: MobileNetV2 base model
        train_ds: Training dataset
        val_ds: Validation dataset
        output_path: Path to save trained model
        
    Returns:
        Training history
    """
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    histories = []
    
    # ========================================
    # PHASE 1: Train classification head
    # ========================================
    print("\n" + "=" * 70)
    print("📚 PHASE 1: Training Classification Head")
    print("   Base model: FROZEN")
    print(f"   Learning rate: {LEARNING_RATE_PHASE1}")
    print(f"   Epochs: {PHASE1_EPOCHS}")
    print("=" * 70)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    history1 = model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    histories.append(history1)
    
    # ========================================
    # PHASE 2: Fine-tune top layers
    # ========================================
    print("\n" + "=" * 70)
    print("🔧 PHASE 2: Fine-tuning Top Layers")
    print("   Unfreezing last 50 layers of MobileNetV2")
    print("   Learning rate: Cosine decay from 1e-4")
    print(f"   Epochs: {PHASE2_EPOCHS}")
    print("=" * 70)
    
    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"   Unfroze {trainable_count} layers")
    
    # Cosine decay learning rate
    total_steps = len(train_ds) * PHASE2_EPOCHS
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=total_steps,
        alpha=0.01
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_ds,
        epochs=PHASE2_EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    histories.append(history2)
    
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
        verbose=1
    )
    histories.append(history3)
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"\n💾 Model saved to: {output_path}")
    
    return histories

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model: keras.Model, test_ds) -> dict:
    """
    Evaluate the trained model on test dataset.
    
    Args:
        model: Trained Keras model
        test_ds: Test dataset
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 70)
    print("📊 EVALUATING ON TEST SET")
    print("=" * 70)
    
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    
    print(f"\n   Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    }

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
        train_ds, val_ds, test_ds = load_dataset(args.data_path, BATCH_SIZE)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Create data augmentation
    data_augmentation = create_data_augmentation()
    
    # Build model
    model, base_model = build_model(data_augmentation)
    model.summary()
    
    # Train model
    histories = train_model(model, base_model, train_ds, val_ds, args.output)
    
    # Evaluate model
    metrics = evaluate_model(model, test_ds)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📈 TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Final Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"   Model saved to: {args.output}")
    print("=" * 70)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
