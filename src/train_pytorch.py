#!/usr/bin/env python3
"""
Accident Detection Model Training - PyTorch Version
====================================================

GPU-accelerated training using PyTorch with MobileNetV2 transfer learning.
Optimized for NVIDIA RTX GPUs with CUDA support.

Three-phase progressive fine-tuning:
    Phase 1: Train classification head (base frozen)
    Phase 2: Fine-tune top layers
    Phase 3: Polish entire network

Author: Accident Detection Research Team
Date: December 2024
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Image settings
IMG_SIZE = 224

# Training settings
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to 0 for Windows compatibility

# Phase epochs
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 15
PHASE3_EPOCHS = 5

# Learning rates
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-4
LEARNING_RATE_PHASE3 = 1e-5

# Warmup
WARMUP_EPOCHS = 2

# Regularization
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-4

# Logging
LOG_DIR = "logs"
METRICS_FILE = "training_metrics.json"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATA LOADING
# ============================================================================

def get_data_transforms():
    """
    Create data augmentation and preprocessing transforms.
    
    Returns:
        Dictionary with train, val, and test transforms
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.1),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def load_datasets(data_path: str, batch_size: int = BATCH_SIZE):
    """
    Load train, validation, and test datasets.
    
    Args:
        data_path: Path to dataset directory
        batch_size: Batch size for data loaders
        
    Returns:
        Data loaders and class names
    """
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    test_dir = os.path.join(data_path, "test")
    
    # Validate directories
    for dir_path, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{name} directory not found: {dir_path}")
    
    print(f"\n📂 Loading dataset from: {data_path}")
    
    transforms_dict = get_data_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transforms_dict['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=transforms_dict['val'])
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms_dict['test'])
    
    # Get class names
    class_names = train_dataset.classes
    print(f"   Classes: {class_names}")
    
    # Count samples per class
    train_counts = {}
    for _, label in train_dataset.samples:
        class_name = class_names[label]
        train_counts[class_name] = train_counts.get(class_name, 0) + 1
    
    total_train = len(train_dataset)
    print(f"   Train samples: {total_train} ({', '.join(f'{k}: {v}' for k, v in train_counts.items())})")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Calculate class weights for balanced training
    class_weights = torch.tensor([
        total_train / (2 * train_counts[class_names[0]]),
        total_train / (2 * train_counts[class_names[1]])
    ], dtype=torch.float32).to(DEVICE)
    print(f"   Class weights: {class_weights.tolist()}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights, class_names


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class AccidentDetector(nn.Module):
    """
    Accident Detection Model using MobileNetV2 backbone.
    
    Architecture:
        MobileNetV2 (pre-trained on ImageNet)
        → Global Average Pooling
        → Dropout(0.5)
        → Dense(512) → BatchNorm → ReLU → Dropout(0.4)
        → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
        → Dense(128) → BatchNorm → ReLU → Dropout(0.2)
        → Dense(1) → Sigmoid
    """
    
    def __init__(self, pretrained: bool = True):
        super(AccidentDetector, self).__init__()
        
        # Load MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Get the number of features from backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1),
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        """Freeze all backbone layers (classifier remains trainable)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Ensure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
            
    def unfreeze_top_layers(self, num_layers: int = 50):
        """Unfreeze top N layers of backbone."""
        # Get all parameters
        params = list(self.backbone.parameters())
        total_params = len(params)
        
        # Freeze all first
        for param in params:
            param.requires_grad = False
        
        # Unfreeze top layers
        for param in params[-num_layers:]:
            param.requires_grad = True
            
        return sum(1 for p in params if p.requires_grad)
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True


def build_model() -> nn.Module:
    """
    Build and initialize the accident detection model.
    
    Returns:
        PyTorch model on the appropriate device
    """
    print("\n🏗️ Building model...")
    
    model = AccidentDetector(pretrained=True)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, phase_name, log_dir, best_acc=0.0, model_path=None):
    """
    Train for one phase with early stopping and checkpointing.
    
    Returns:
        Training history and best accuracy
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    patience = 8
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, epochs
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"   Epoch {epoch}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            if model_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, model_path.replace('.pth', '_best.pth'))
                print(f"   ✓ Saved best model (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping triggered at epoch {epoch}")
                break
    
    return history, best_acc


def train_model(model, train_loader, val_loader, output_path: str, class_weights):
    """
    Train the model using three-phase progressive fine-tuning.
    
    Phase 1: Train classification head (base frozen)
    Phase 2: Fine-tune top 50 layers
    Phase 3: Polish entire network
    """
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOG_DIR, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Loss function with label smoothing
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])
    
    all_metrics = {'phases': [], 'timestamp': timestamp}
    best_acc = 0.0
    
    # ========================================
    # PHASE 1: Train classification head
    # ========================================
    print("\n" + "=" * 70)
    print("📚 PHASE 1: Training Classification Head")
    print("   Base model: FROZEN")
    print(f"   Learning rate: {LEARNING_RATE_PHASE1}")
    print(f"   Epochs: {PHASE1_EPOCHS}")
    print("=" * 70)
    
    model.freeze_backbone()
    
    # Only optimize classifier parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_PHASE1,
        weight_decay=WEIGHT_DECAY
    )
    
    # Warmup + Cosine annealing
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS * len(train_loader))
    main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=PHASE1_EPOCHS - WARMUP_EPOCHS)
    
    history1, best_acc = train_phase(
        model, train_loader, val_loader, criterion, optimizer, warmup_scheduler,
        PHASE1_EPOCHS, "Phase 1", log_dir, best_acc, output_path
    )
    
    all_metrics['phases'].append({
        'phase': 1,
        'name': 'Classification Head Training',
        'epochs_completed': len(history1['train_loss']),
        'final_train_acc': history1['train_acc'][-1],
        'final_val_acc': history1['val_acc'][-1],
        'best_val_acc': max(history1['val_acc'])
    })
    
    # ========================================
    # PHASE 2: Fine-tune top layers
    # ========================================
    print("\n" + "=" * 70)
    print("🔧 PHASE 2: Fine-tuning Top Layers")
    print("   Unfreezing last 50 layers of MobileNetV2")
    print(f"   Learning rate: {LEARNING_RATE_PHASE2}")
    print(f"   Epochs: {PHASE2_EPOCHS}")
    print("=" * 70)
    
    unfrozen = model.unfreeze_top_layers(50)
    print(f"   Unfroze {unfrozen} parameters")
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_PHASE2,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=PHASE2_EPOCHS)
    
    history2, best_acc = train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        PHASE2_EPOCHS, "Phase 2", log_dir, best_acc, output_path
    )
    
    all_metrics['phases'].append({
        'phase': 2,
        'name': 'Fine-tuning Top Layers',
        'epochs_completed': len(history2['train_loss']),
        'final_train_acc': history2['train_acc'][-1],
        'final_val_acc': history2['val_acc'][-1],
        'best_val_acc': max(history2['val_acc'])
    })
    
    # ========================================
    # PHASE 3: Final polish
    # ========================================
    print("\n" + "=" * 70)
    print("🎯 PHASE 3: Final Polish")
    print("   All layers: TRAINABLE")
    print(f"   Learning rate: {LEARNING_RATE_PHASE3}")
    print(f"   Epochs: {PHASE3_EPOCHS}")
    print("=" * 70)
    
    model.unfreeze_all()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_PHASE3,
        weight_decay=WEIGHT_DECAY
    )
    
    history3, best_acc = train_phase(
        model, train_loader, val_loader, criterion, optimizer, None,
        PHASE3_EPOCHS, "Phase 3", log_dir, best_acc, output_path
    )
    
    all_metrics['phases'].append({
        'phase': 3,
        'name': 'Final Polish',
        'epochs_completed': len(history3['train_loss']),
        'final_train_acc': history3['train_acc'][-1],
        'final_val_acc': history3['val_acc'][-1],
        'best_val_acc': max(history3['val_acc'])
    })
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': ['Accident', 'Non Accident'],
    }, output_path)
    print(f"\n💾 Model saved to: {output_path}")
    
    # Save training metrics
    metrics_path = os.path.join(log_dir, METRICS_FILE)
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"📊 Training metrics saved to: {metrics_path}")
    
    return [history1, history2, history3], all_metrics, log_dir, best_acc


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, class_names, log_dir=None):
    """
    Evaluate the trained model on test dataset with comprehensive metrics.
    """
    print("\n" + "=" * 70)
    print("📊 EVALUATING ON TEST SET")
    print("=" * 70)
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels_gpu = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels_gpu)
            total_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    test_loss = total_loss / len(test_loader.dataset)
    test_acc = (y_true == y_pred).mean()
    
    print(f"\n   Test Accuracy: {test_acc * 100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n   📈 CONFUSION MATRIX:")
    print(f"   {'':<15} Predicted")
    print(f"   {'':<15} {class_names[0]:<12} {class_names[1]:<12}")
    print(f"   Actual {class_names[0]:<6} {cm[0][0]:<12} {cm[0][1]:<12}")
    print(f"   Actual {class_names[1]:<6} {cm[1][0]:<12} {cm[1][1]:<12}")
    
    # Calculate detailed metrics
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
        'test_accuracy': float(test_acc),
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
        description='Train Accident Detection Model (PyTorch GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_pytorch.py --data_path /path/to/dataset
  python train_pytorch.py --data_path /path/to/dataset --output models/model.pth
        """
    )
    
    parser.add_argument('--data_path', '-d', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output', '-o', type=str, 
                        default='models/accident_detector.pth',
                        help='Path to save trained model')
    parser.add_argument('--batch_size', '-b', type=int, default=BATCH_SIZE,
                        help='Training batch size (default: 32)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚗 ACCIDENT DETECTION MODEL TRAINING (PyTorch GPU)")
    print("   Transfer Learning with MobileNetV2")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
    else:
        print("\n⚠️ No GPU detected, training on CPU (slower)")
    
    # Load dataset
    try:
        train_loader, val_loader, test_loader, class_weights, class_names = load_datasets(
            args.data_path, args.batch_size
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Build model
    model = build_model()
    
    # Print model summary
    print("\n📋 Model Architecture:")
    print(model)
    
    # Train model
    histories, training_metrics, log_dir, best_acc = train_model(
        model, train_loader, val_loader, args.output, class_weights
    )
    
    # Load best model for evaluation
    best_model_path = args.output.replace('.pth', '_best.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n📂 Loaded best model from: {best_model_path}")
    
    # Evaluate model
    eval_metrics = evaluate_model(model, test_loader, class_names, log_dir)
    
    # Save complete metrics
    complete_metrics = {
        'training': training_metrics,
        'evaluation': eval_metrics,
        'config': {
            'batch_size': args.batch_size,
            'phase1_epochs': PHASE1_EPOCHS,
            'phase2_epochs': PHASE2_EPOCHS,
            'phase3_epochs': PHASE3_EPOCHS,
            'warmup_epochs': WARMUP_EPOCHS,
            'label_smoothing': LABEL_SMOOTHING,
            'weight_decay': WEIGHT_DECAY,
            'dropout_rate': DROPOUT_RATE,
            'device': str(DEVICE),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
    }
    
    final_metrics_path = os.path.join(log_dir, 'complete_training_report.json')
    with open(final_metrics_path, 'w') as f:
        json.dump(complete_metrics, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📈 TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Best Val Accuracy:   {best_acc * 100:.2f}%")
    print(f"   Final Test Accuracy: {eval_metrics['test_accuracy'] * 100:.2f}%")
    print(f"   Precision:           {eval_metrics['precision'] * 100:.2f}%")
    print(f"   Recall:              {eval_metrics['recall'] * 100:.2f}%")
    print(f"   F1-Score:            {eval_metrics['f1_score'] * 100:.2f}%")
    print(f"   Model saved to:      {args.output}")
    print(f"   Best model:          {best_model_path}")
    print(f"   Logs saved to:       {log_dir}")
    print("=" * 70)
    
    print("\n✅ Training complete! Ready for research paper documentation.")


if __name__ == "__main__":
    main()
