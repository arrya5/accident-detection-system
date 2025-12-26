"""
Accident Detection System - Model Verification Module (PyTorch)

This module verifies the trained model to detect potential overfitting
and provides detailed analysis of model performance on unseen data.

Author: Arya Bhardwaj
Date: December 2025
License: MIT

Features:
    - Tests model on validation and test datasets
    - Analyzes confidence distributions per class
    - Detects potential overfitting indicators
    - Generates detailed verification report
    - Exports report to file (optional)
    - ROC-AUC curve generation for research papers
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import os
import argparse
import sys
import json
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# Optional: matplotlib for ROC curve
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows compatibility

# Overfitting thresholds
OVERFIT_ACCURACY_THRESHOLD = 0.98  # Flag if accuracy > 98%
MIN_CONFIDENCE_VARIANCE = 0.01     # Flag if predictions too uniform
CLASS_IMBALANCE_THRESHOLD = 0.2    # Flag if class accuracy differs by > 20%


# ============================================================================
# MODEL DEFINITION (must match training and detect_pytorch.py)
# ============================================================================

class AccidentDetector(nn.Module):
    """MobileNetV2-based accident detection model."""
    
    def __init__(self, num_classes=1, pretrained=False):
        super(AccidentDetector, self).__init__()
        
        # Load MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        # Get the number of features from backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with Identity (we'll use our own)
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
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load the trained PyTorch model."""
    default_paths = [
        "models/accident_detector_best.pth",
        "models/accident_detector_final.pth",
        "../models/accident_detector_best.pth",
        "../models/accident_detector_final.pth",
    ]
    
    paths_to_try = [model_path] if model_path else default_paths
    
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"✅ Loading model from: {path}")
            
            # Create model with matching architecture
            model = AccidentDetector(num_classes=1, pretrained=False)
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            return model
    
    raise FileNotFoundError(f"Model not found. Searched: {paths_to_try}")


# ============================================================================
# DATA LOADING
# ============================================================================

def get_transforms():
    """Get inference transforms."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def load_dataset(data_path: str, split: str = "test"):
    """Load a dataset split."""
    split_dir = os.path.join(data_path, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Directory not found: {split_dir}")
    
    dataset = ImageFolder(split_dir, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                       shuffle=False, num_workers=NUM_WORKERS)
    
    return dataset, loader


# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_on_dataset(model: nn.Module, loader: DataLoader, 
                      class_names: list, device: torch.device,
                      desc: str = "Verifying") -> dict:
    """
    Verify model performance on a dataset.
    
    Returns:
        Dictionary with detailed verification results
    """
    model.eval()
    
    # Storage for predictions and labels
    all_preds = []
    all_labels = []
    all_confidences = []
    all_raw_probs = []  # For ROC curve
    
    # Per-class storage
    class_confidences = defaultdict(list)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            
            # Handle single sample batch
            if len(probs.shape) == 0:
                probs = probs.unsqueeze(0)
            
            # Class 0 = Accident, Class 1 = Non-Accident
            # P(Accident) = 1 - sigmoid(output)
            accident_probs = 1 - probs
            
            # Predictions
            preds = (accident_probs >= 0.5).long()
            
            # Store results
            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                conf = accident_probs[i].item()
                
                all_preds.append(pred)
                all_labels.append(label)
                all_raw_probs.append(conf)  # Store raw accident probability
                all_confidences.append(conf if label == 0 else 1 - conf)
                
                # Per-class tracking
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                
                # Store confidence for the true class
                if label == 0:  # Accident
                    class_confidences[0].append(conf)
                else:  # Non-Accident
                    class_confidences[1].append(1 - conf)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_raw_probs = np.array(all_raw_probs)
    
    total = len(all_labels)
    correct = np.sum(all_preds == all_labels)
    accuracy = correct / total
    
    # Per-class accuracy
    class_accuracy = {}
    for cls in class_total:
        class_accuracy[cls] = class_correct[cls] / class_total[cls]
    
    # Confidence statistics
    confidence_stats = {
        'mean': float(np.mean(all_confidences)),
        'std': float(np.std(all_confidences)),
        'min': float(np.min(all_confidences)),
        'max': float(np.max(all_confidences)),
        'median': float(np.median(all_confidences))
    }
    
    # Per-class confidence stats
    per_class_conf_stats = {}
    for cls in class_confidences:
        confs = np.array(class_confidences[cls])
        per_class_conf_stats[cls] = {
            'mean': float(np.mean(confs)),
            'std': float(np.std(confs)),
            'min': float(np.min(confs)),
            'max': float(np.max(confs))
        }
    
    # Error analysis
    false_positives = int(np.sum((all_preds == 0) & (all_labels == 1)))
    false_negatives = int(np.sum((all_preds == 1) & (all_labels == 0)))
    true_positives = int(np.sum((all_preds == 0) & (all_labels == 0)))
    true_negatives = int(np.sum((all_preds == 1) & (all_labels == 1)))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # ROC-AUC (invert labels and probs for proper calculation)
    # Label 0 = Accident (positive), Label 1 = Non-Accident (negative)
    binary_labels = 1 - all_labels  # Convert: Accident=1, Non-Accident=0
    roc_auc = 0.0
    pr_auc = 0.0
    
    if PLOTTING_AVAILABLE:
        try:
            fpr, tpr, _ = roc_curve(binary_labels, all_raw_probs)
            roc_auc = float(auc(fpr, tpr))
            
            precision_curve, recall_curve, _ = precision_recall_curve(binary_labels, all_raw_probs)
            pr_auc = float(auc(recall_curve, precision_curve))
        except:
            pass
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': float(accuracy),
        'class_accuracy': {int(k): float(v) for k, v in class_accuracy.items()},
        'class_total': {int(k): int(v) for k, v in class_total.items()},
        'class_correct': {int(k): int(v) for k, v in class_correct.items()},
        'confidence_stats': confidence_stats,
        'per_class_conf_stats': {int(k): v for k, v in per_class_conf_stats.items()},
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        # Store for plotting
        '_labels': binary_labels.tolist(),
        '_probs': all_raw_probs.tolist()
    }


def check_overfitting(val_results: dict, test_results: dict) -> list:
    """
    Check for potential overfitting indicators.
    
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Check 1: Extremely high accuracy (might indicate data leakage)
    if val_results['accuracy'] > OVERFIT_ACCURACY_THRESHOLD:
        warnings.append(
            f"⚠️ Validation accuracy ({val_results['accuracy']*100:.2f}%) is very high. "
            f"Verify no data leakage between train/val sets."
        )
    
    if test_results['accuracy'] > OVERFIT_ACCURACY_THRESHOLD:
        warnings.append(
            f"⚠️ Test accuracy ({test_results['accuracy']*100:.2f}%) is very high. "
            f"Verify no data leakage between train/test sets."
        )
    
    # Check 2: Large gap between val and test accuracy
    acc_gap = abs(val_results['accuracy'] - test_results['accuracy'])
    if acc_gap > 0.05:
        warnings.append(
            f"⚠️ Accuracy gap between val ({val_results['accuracy']*100:.2f}%) and "
            f"test ({test_results['accuracy']*100:.2f}%) is {acc_gap*100:.2f}%. "
            f"This might indicate overfitting to validation set."
        )
    
    # Check 3: Low confidence variance (model too certain)
    if val_results['confidence_stats']['std'] < MIN_CONFIDENCE_VARIANCE:
        warnings.append(
            f"⚠️ Confidence variance on validation set is very low "
            f"({val_results['confidence_stats']['std']:.4f}). "
            f"Model predictions might be too uniform."
        )
    
    # Check 4: Class imbalance in accuracy
    val_class_acc = val_results['class_accuracy']
    if len(val_class_acc) >= 2:
        acc_diff = abs(val_class_acc.get(0, 0) - val_class_acc.get(1, 0))
        if acc_diff > CLASS_IMBALANCE_THRESHOLD:
            warnings.append(
                f"⚠️ Class accuracy imbalance detected. "
                f"Accident: {val_class_acc.get(0, 0)*100:.2f}%, "
                f"Non-Accident: {val_class_acc.get(1, 0)*100:.2f}%. "
                f"Model may be biased toward one class."
            )
    
    # Check 5: Zero false positives or false negatives (suspicious)
    if test_results['false_positives'] == 0 and test_results['false_negatives'] == 0:
        warnings.append(
            f"⚠️ Perfect classification with zero errors. "
            f"This is unusual - verify test set is truly unseen data."
        )
    
    return warnings


def plot_roc_curve(test_results: dict, output_dir: str = "."):
    """Generate and save ROC curve plot."""
    if not PLOTTING_AVAILABLE:
        print("   ⚠️ matplotlib/sklearn not installed - skipping ROC curve")
        return None
    
    labels = np.array(test_results['_labels'])
    probs = np.array(test_results['_probs'])
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'roc_pr_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 ROC & PR curves saved to: {output_path}")
    return output_path


def print_verification_report(val_results: dict, test_results: dict, 
                              class_names: list, warnings: list):
    """Print a detailed verification report."""
    print("\n" + "=" * 70)
    print("🔍 MODEL VERIFICATION REPORT")
    print("=" * 70)
    
    # Validation Results
    print("\n📊 VALIDATION SET RESULTS")
    print("-" * 40)
    print(f"   Total Samples: {val_results['total']}")
    print(f"   Correct: {val_results['correct']}")
    print(f"   Accuracy: {val_results['accuracy']*100:.2f}%")
    print(f"\n   Per-Class Performance:")
    for cls in val_results['class_accuracy']:
        name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        acc = val_results['class_accuracy'][cls]
        total = val_results['class_total'][cls]
        correct = val_results['class_correct'][cls]
        print(f"      {name}: {correct}/{total} ({acc*100:.2f}%)")
    
    # Test Results
    print("\n📊 TEST SET RESULTS")
    print("-" * 40)
    print(f"   Total Samples: {test_results['total']}")
    print(f"   Correct: {test_results['correct']}")
    print(f"   Accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"\n   Per-Class Performance:")
    for cls in test_results['class_accuracy']:
        name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        acc = test_results['class_accuracy'][cls]
        total = test_results['class_total'][cls]
        correct = test_results['class_correct'][cls]
        print(f"      {name}: {correct}/{total} ({acc*100:.2f}%)")
    
    # Confusion Matrix
    print("\n📈 CONFUSION MATRIX (Test Set)")
    print("-" * 40)
    print(f"                    Predicted")
    print(f"                    Accident    Normal")
    print(f"   Actual Accident    {test_results['true_positives']:4d}       {test_results['false_negatives']:4d}")
    print(f"   Actual Normal      {test_results['false_positives']:4d}       {test_results['true_negatives']:4d}")
    
    # Metrics
    print("\n📏 METRICS (Test Set)")
    print("-" * 40)
    print(f"   Accuracy:    {test_results['accuracy']*100:.2f}%")
    print(f"   Precision:   {test_results['precision']*100:.2f}%")
    print(f"   Recall:      {test_results['recall']*100:.2f}%")
    print(f"   F1-Score:    {test_results['f1_score']*100:.2f}%")
    print(f"   Specificity: {test_results['specificity']*100:.2f}%")
    if test_results['roc_auc'] > 0:
        print(f"   ROC-AUC:     {test_results['roc_auc']:.4f}")
        print(f"   PR-AUC:      {test_results['pr_auc']:.4f}")
    
    # Confidence Analysis
    print("\n🎯 CONFIDENCE ANALYSIS")
    print("-" * 40)
    print(f"   Overall Confidence Stats:")
    stats = test_results['confidence_stats']
    print(f"      Mean:   {stats['mean']*100:.2f}%")
    print(f"      Std:    {stats['std']*100:.2f}%")
    print(f"      Min:    {stats['min']*100:.2f}%")
    print(f"      Max:    {stats['max']*100:.2f}%")
    print(f"      Median: {stats['median']*100:.2f}%")
    
    print(f"\n   Per-Class Confidence:")
    for cls in test_results['per_class_conf_stats']:
        name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        cls_stats = test_results['per_class_conf_stats'][cls]
        print(f"      {name}: Mean={cls_stats['mean']*100:.2f}%, Std={cls_stats['std']*100:.2f}%")
    
    # Overfitting Warnings
    print("\n⚠️ OVERFITTING CHECK")
    print("-" * 40)
    if warnings:
        for warning in warnings:
            print(f"   {warning}")
    else:
        print("   ✅ No overfitting indicators detected!")
    
    # Final Verdict
    print("\n" + "=" * 70)
    if not warnings and test_results['accuracy'] >= 0.95:
        print("✅ VERDICT: Model passes verification - ready for deployment!")
    elif warnings:
        print("⚠️ VERDICT: Review warnings above before deployment.")
    else:
        print(f"ℹ️ VERDICT: Model accuracy ({test_results['accuracy']*100:.2f}%) - may need improvement.")
    print("=" * 70)


def export_report(val_results: dict, test_results: dict, 
                  class_names: list, warnings: list, output_path: str):
    """Export verification report to JSON file."""
    # Remove internal plotting data
    val_clean = {k: v for k, v in val_results.items() if not k.startswith('_')}
    test_clean = {k: v for k, v in test_results.items() if not k.startswith('_')}
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'validation_results': val_clean,
        'test_results': test_clean,
        'class_names': class_names,
        'warnings': warnings,
        'verdict': 'PASS' if (not warnings and test_results['accuracy'] >= 0.95) else 
                   'REVIEW' if warnings else 'NEEDS_IMPROVEMENT'
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report exported to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for model verification."""
    parser = argparse.ArgumentParser(
        description='Verify PyTorch Accident Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_model_pytorch.py --data_path data
  python verify_model_pytorch.py --data_path data --model models/accident_detector_best.pth
  python verify_model_pytorch.py --data_path data --export --plot
        """
    )
    
    parser.add_argument('--data_path', '-d', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file (.pth)')
    parser.add_argument('--export', '-e', action='store_true',
                        help='Export report to JSON file')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Generate ROC and PR curve plots')
    parser.add_argument('--output_dir', '-o', type=str, default='output',
                        help='Output directory for exports (default: output)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 MODEL VERIFICATION")
    print("   Checking for overfitting and validating performance")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load model
    try:
        model = load_model(args.model, device)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Load datasets
    try:
        print("\n📂 Loading datasets...")
        val_dataset, val_loader = load_dataset(args.data_path, "val")
        test_dataset, test_loader = load_dataset(args.data_path, "test")
        class_names = val_dataset.classes
        print(f"   Classes: {class_names}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Verify on validation set
    print("\n🔄 Verifying on validation set...")
    val_results = verify_on_dataset(model, val_loader, class_names, device, "Validation")
    
    # Verify on test set
    print("🔄 Verifying on test set...")
    test_results = verify_on_dataset(model, test_loader, class_names, device, "Test")
    
    # Check for overfitting
    warnings = check_overfitting(val_results, test_results)
    
    # Print report
    print_verification_report(val_results, test_results, class_names, warnings)
    
    # Export options
    if args.export or args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.export:
            export_path = os.path.join(args.output_dir, 'verification_report.json')
            export_report(val_results, test_results, class_names, warnings, export_path)
        
        if args.plot:
            plot_roc_curve(test_results, args.output_dir)
    
    print("\n✅ Verification complete!")


if __name__ == "__main__":
    main()
