"""
Accident Detection System - Interactive Dataset Testing Module (PyTorch)

This module allows testing the trained model on random samples
from the test dataset with interactive visual feedback.

Author: Arya Bhardwaj
Date: December 2025
License: MIT

Features:
    - Random sample testing from val/test sets
    - Interactive keyboard navigation
    - Visual overlay with predictions and confidence
    - Per-class accuracy breakdown
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
import os
import random
import argparse
import sys
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 224

# ============================================================================
# MODEL LOADING
# ============================================================================

def build_model():
    """Build the model architecture (must match training)."""
    base_model = models.mobilenet_v2(weights=None)
    
    # Custom classifier head (same as training)
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    )
    
    return base_model


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
            model = build_model()
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            return model
    
    raise FileNotFoundError(f"Model not found. Searched: {paths_to_try}")


# ============================================================================
# INFERENCE
# ============================================================================

def get_transforms():
    """Get inference transforms."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def predict_image(model: nn.Module, img_path: str, 
                  device: torch.device, transform) -> tuple:
    """
    Predict accident probability for an image.
    
    Returns:
        Tuple of (accident_prob, is_accident)
    """
    # Load and preprocess
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    
    # Class 0 = Accident, Class 1 = Non-Accident
    # P(Accident) = 1 - sigmoid(output)
    accident_prob = 1 - prob
    is_accident = accident_prob >= 0.5
    
    return accident_prob, is_accident


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_display(img: np.ndarray, true_label: str, is_correct: bool,
                   confidence: float, idx: int, total: int) -> np.ndarray:
    """Create display with prediction overlay."""
    display = img.copy()
    h, w = display.shape[:2]
    
    # Resize if too large
    max_width = 800
    if w > max_width:
        scale = max_width / w
        display = cv2.resize(display, (max_width, int(h * scale)))
        h, w = display.shape[:2]
    
    # Border color (green = correct, red = wrong)
    color = (0, 255, 0) if is_correct else (0, 0, 255)
    cv2.rectangle(display, (0, 0), (w-1, h-1), color, 10)
    
    # Create info panel
    panel_height = 130
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    
    # True label
    cv2.putText(panel, f"True Label: {true_label} ({idx}/{total})", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Prediction result
    result_text = "CORRECT" if is_correct else "WRONG"
    result_symbol = "[OK]" if is_correct else "[X]"
    cv2.putText(panel, f"Prediction: {result_symbol} {result_text}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Confidence bar
    bar_w = 300
    bar_x = 20
    bar_y = 90
    bar_h = 25
    
    # Background
    cv2.rectangle(panel, (bar_x, bar_y), 
                  (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    
    # Fill
    fill_w = int(bar_w * confidence)
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + bar_h), color, -1)
    
    # Border
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
    
    # Confidence text
    cv2.putText(panel, f"Confidence: {confidence*100:.1f}%",
                (bar_x + bar_w + 20, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Combine panel and image
    display = np.vstack([panel, display])
    
    # Add instructions at bottom
    instr_h = 35
    instr_panel = np.zeros((instr_h, display.shape[1], 3), dtype=np.uint8)
    instr_panel[:] = (30, 30, 30)
    cv2.putText(instr_panel, "Press ANY KEY for next | Q to quit", 
                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    display = np.vstack([display, instr_panel])
    
    return display


# ============================================================================
# TESTING
# ============================================================================

def test_on_dataset(model: nn.Module, data_path: str, 
                    num_samples: int, device: torch.device) -> dict:
    """
    Test model on random samples from the dataset.
    
    Args:
        model: Trained PyTorch model
        data_path: Path to dataset root
        num_samples: Number of samples to test per class
        device: PyTorch device
        
    Returns:
        Dictionary with test results
    """
    transform = get_transforms()
    
    # Find test directory
    test_dir = os.path.join(data_path, "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_path, "val")
    
    accident_dir = os.path.join(test_dir, "Accident")
    normal_dir = os.path.join(test_dir, "Non Accident")
    
    if not os.path.exists(accident_dir) or not os.path.exists(normal_dir):
        raise FileNotFoundError(f"Test directories not found in: {test_dir}")
    
    # Get image files
    accident_images = [f for f in os.listdir(accident_dir) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    normal_images = [f for f in os.listdir(normal_dir) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Random sample
    sample_accidents = random.sample(accident_images, 
                                      min(num_samples, len(accident_images)))
    sample_normals = random.sample(normal_images, 
                                    min(num_samples, len(normal_images)))
    
    print(f"\n📂 Testing on {len(sample_accidents)} accident + {len(sample_normals)} normal images")
    print("   Press any key for next image, 'Q' to quit\n")
    
    # Statistics
    accident_correct = 0
    normal_correct = 0
    quit_early = False
    
    # Test accident images
    print("🔴 Testing ACCIDENT images...")
    for idx, img_name in enumerate(sample_accidents, 1):
        img_path = os.path.join(accident_dir, img_name)
        
        # Load image for display
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Predict
        accident_prob, is_accident = predict_image(model, img_path, device, transform)
        
        is_correct = is_accident  # Should predict as accident
        if is_correct:
            accident_correct += 1
        
        # Create display
        display = create_display(img, "ACCIDENT", is_correct, 
                                 accident_prob, idx, len(sample_accidents))
        
        cv2.imshow('Dataset Testing - PyTorch', display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == ord('Q'):
            quit_early = True
            break
    
    # Test normal images
    if not quit_early:
        print("🟢 Testing NORMAL images...")
        for idx, img_name in enumerate(sample_normals, 1):
            img_path = os.path.join(normal_dir, img_name)
            
            # Load image for display
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Predict
            accident_prob, is_accident = predict_image(model, img_path, device, transform)
            normal_prob = 1 - accident_prob
            
            is_correct = not is_accident  # Should predict as normal
            if is_correct:
                normal_correct += 1
            
            # Create display
            display = create_display(img, "NORMAL", is_correct, 
                                     normal_prob, idx, len(sample_normals))
            
            cv2.imshow('Dataset Testing - PyTorch', display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
    
    cv2.destroyAllWindows()
    
    # Calculate results
    total_tested = len(sample_accidents) + len(sample_normals)
    total_correct = accident_correct + normal_correct
    
    results = {
        'total_tested': total_tested,
        'total_correct': total_correct,
        'accuracy': total_correct / total_tested * 100 if total_tested > 0 else 0,
        'accident_correct': accident_correct,
        'accident_total': len(sample_accidents),
        'normal_correct': normal_correct,
        'normal_total': len(sample_normals),
    }
    
    return results


def print_results(results: dict):
    """Print test results summary."""
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"   Total Tested: {results['total_tested']}")
    print(f"   Correct: {results['total_correct']}")
    print(f"   Accuracy: {results['accuracy']:.1f}%")
    print(f"\n   Accident: {results['accident_correct']}/{results['accident_total']} "
          f"({results['accident_correct']/results['accident_total']*100:.1f}%)")
    print(f"   Normal: {results['normal_correct']}/{results['normal_total']} "
          f"({results['normal_correct']/results['normal_total']*100:.1f}%)")
    print("=" * 50)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for dataset testing."""
    parser = argparse.ArgumentParser(
        description='Interactive Dataset Testing (PyTorch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_dataset_pytorch.py --data_path data
  python test_dataset_pytorch.py --data_path data --samples 20
  python test_dataset_pytorch.py --data_path data --model models/accident_detector_best.pth
        """
    )
    
    parser.add_argument('--data_path', '-d', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file (.pth)')
    parser.add_argument('--samples', '-s', type=int, default=10,
                        help='Number of samples per class (default: 10)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🧪 INTERACTIVE DATASET TESTING")
    print("   Testing on random samples with visual feedback")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    try:
        model = load_model(args.model, device)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Run tests
    try:
        results = test_on_dataset(model, args.data_path, args.samples, device)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Print results
    print_results(results)
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
