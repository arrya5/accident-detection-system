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
    - Optional TTA (Test-Time Augmentation)
    - Auto-cycle mode for presentations
    - Save incorrect predictions for analysis
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import os
import random
import argparse
import sys
from PIL import Image
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 224

# Colors (BGR for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)


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
        
        # Replace classifier with Identity
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
            
            model = AccidentDetector(num_classes=1, pretrained=False)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            
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


def apply_tta_transforms(img: Image.Image, transform) -> torch.Tensor:
    """Apply Test-Time Augmentation and return batch of tensors."""
    augmented = []
    
    # Original
    augmented.append(transform(img))
    
    # Horizontal flip
    augmented.append(transform(img.transpose(Image.FLIP_LEFT_RIGHT)))
    
    # Slight brightness variations using PIL
    from PIL import ImageEnhance
    
    # Brightness up
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(transform(enhancer.enhance(1.1)))
    
    # Brightness down
    augmented.append(transform(enhancer.enhance(0.9)))
    
    # Contrast
    enhancer = ImageEnhance.Contrast(img)
    augmented.append(transform(enhancer.enhance(1.2)))
    
    return torch.stack(augmented)


def predict_image(model: nn.Module, img_path: str, 
                  device: torch.device, transform,
                  use_tta: bool = False) -> tuple:
    """
    Predict accident probability for an image.
    
    Returns:
        Tuple of (accident_prob, is_accident, individual_preds)
    """
    # Load and preprocess
    img = Image.open(img_path).convert('RGB')
    
    with torch.no_grad():
        if use_tta:
            # Apply TTA
            batch = apply_tta_transforms(img, transform).to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs).squeeze()
            accident_probs = 1 - probs  # P(Accident) = 1 - sigmoid(output)
            accident_prob = accident_probs.mean().item()
            individual = accident_probs.cpu().numpy()
        else:
            # Single prediction
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            accident_prob = 1 - prob
            individual = np.array([accident_prob])
    
    is_accident = accident_prob >= 0.5
    
    return accident_prob, is_accident, individual


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_display(img: np.ndarray, true_label: str, is_correct: bool,
                   confidence: float, idx: int, total: int,
                   use_tta: bool, individual_preds: np.ndarray = None,
                   auto_mode: bool = False) -> np.ndarray:
    """Create display with prediction overlay."""
    display = img.copy()
    h, w = display.shape[:2]
    
    # Resize if too large
    max_width = 900
    if w > max_width:
        scale = max_width / w
        display = cv2.resize(display, (max_width, int(h * scale)))
        h, w = display.shape[:2]
    
    # Border color (green = correct, red = wrong)
    color = COLOR_GREEN if is_correct else COLOR_RED
    cv2.rectangle(display, (0, 0), (w-1, h-1), color, 10)
    
    # Create info panel
    panel_height = 160 if use_tta else 130
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    
    # True label
    cv2.putText(panel, f"True Label: {true_label} ({idx}/{total})", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
    
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    # TTA predictions if available
    if use_tta and individual_preds is not None:
        tta_text = "TTA: " + ", ".join([f"{p*100:.0f}%" for p in individual_preds])
        cv2.putText(panel, tta_text, (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    # Combine panel and image
    display = np.vstack([panel, display])
    
    # Add instructions at bottom
    instr_h = 40
    instr_panel = np.zeros((instr_h, display.shape[1], 3), dtype=np.uint8)
    instr_panel[:] = (30, 30, 30)
    
    if auto_mode:
        instr_text = "AUTO MODE | Press 'Q' to quit | 'M' for manual mode"
    else:
        instr_text = "Press ANY KEY for next | 'A' for auto mode | 'S' to save | 'Q' to quit"
    
    cv2.putText(instr_panel, instr_text, 
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    
    display = np.vstack([display, instr_panel])
    
    return display


# ============================================================================
# TESTING
# ============================================================================

def test_on_dataset(model: nn.Module, data_path: str, 
                    num_samples: int, device: torch.device,
                    use_tta: bool = False, auto_delay: int = 0,
                    save_errors: bool = False, output_dir: str = "output") -> dict:
    """
    Test model on random samples from the dataset.
    
    Args:
        model: Trained PyTorch model
        data_path: Path to dataset root
        num_samples: Number of samples to test per class
        device: PyTorch device
        use_tta: Whether to use Test-Time Augmentation
        auto_delay: Delay in ms for auto mode (0 = manual mode)
        save_errors: Whether to save incorrect predictions
        output_dir: Directory to save error images
        
    Returns:
        Dictionary with test results
    """
    transform = get_transforms()
    auto_mode = auto_delay > 0
    
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
    if use_tta:
        print("   TTA: Enabled (5 augmentations)")
    if auto_mode:
        print(f"   Auto mode: {auto_delay}ms delay")
    print("   Press any key for next image, 'Q' to quit\n")
    
    # Create error output directory if needed
    if save_errors:
        error_dir = os.path.join(output_dir, f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(error_dir, exist_ok=True)
    
    # Statistics
    accident_correct = 0
    normal_correct = 0
    quit_early = False
    errors = []
    
    # Test accident images
    print("🔴 Testing ACCIDENT images...")
    for idx, img_name in enumerate(sample_accidents, 1):
        img_path = os.path.join(accident_dir, img_name)
        
        # Load image for display
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Predict
        accident_prob, is_accident, individual = predict_image(
            model, img_path, device, transform, use_tta
        )
        
        is_correct = is_accident  # Should predict as accident
        if is_correct:
            accident_correct += 1
        else:
            errors.append(('Accident', img_path, accident_prob))
            if save_errors:
                cv2.imwrite(os.path.join(error_dir, f"FN_{img_name}"), img)
        
        # Create display
        display = create_display(img, "ACCIDENT", is_correct, 
                                 accident_prob, idx, len(sample_accidents),
                                 use_tta, individual, auto_mode)
        
        cv2.imshow('Dataset Testing - PyTorch', display)
        
        # Handle key press
        wait_time = auto_delay if auto_mode else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            quit_early = True
            break
        elif key == ord('a') or key == ord('A'):
            auto_mode = True
            auto_delay = 1500
        elif key == ord('m') or key == ord('M'):
            auto_mode = False
            auto_delay = 0
        elif key == ord('s') or key == ord('S'):
            save_path = f"saved_{datetime.now().strftime('%H%M%S')}_{img_name}"
            cv2.imwrite(save_path, display)
            print(f"   💾 Saved: {save_path}")
    
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
            accident_prob, is_accident, individual = predict_image(
                model, img_path, device, transform, use_tta
            )
            normal_prob = 1 - accident_prob
            
            is_correct = not is_accident  # Should predict as normal
            if is_correct:
                normal_correct += 1
            else:
                errors.append(('Normal', img_path, accident_prob))
                if save_errors:
                    cv2.imwrite(os.path.join(error_dir, f"FP_{img_name}"), img)
            
            # Create display
            display = create_display(img, "NORMAL", is_correct, 
                                     normal_prob, idx, len(sample_normals),
                                     use_tta, 1 - individual if individual is not None else None,
                                     auto_mode)
            
            cv2.imshow('Dataset Testing - PyTorch', display)
            
            # Handle key press
            wait_time = auto_delay if auto_mode else 0
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('a') or key == ord('A'):
                auto_mode = True
                auto_delay = 1500
            elif key == ord('m') or key == ord('M'):
                auto_mode = False
                auto_delay = 0
            elif key == ord('s') or key == ord('S'):
                save_path = f"saved_{datetime.now().strftime('%H%M%S')}_{img_name}"
                cv2.imwrite(save_path, display)
                print(f"   💾 Saved: {save_path}")
    
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
        'errors': errors
    }
    
    if save_errors and errors:
        print(f"\n📁 Error images saved to: {error_dir}")
    
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
    
    if results['errors']:
        print(f"\n   ❌ Errors ({len(results['errors'])}):")
        for label, path, conf in results['errors'][:5]:  # Show first 5
            print(f"      {label}: {os.path.basename(path)} (conf: {conf*100:.1f}%)")
        if len(results['errors']) > 5:
            print(f"      ... and {len(results['errors']) - 5} more")
    
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
  python test_dataset_pytorch.py --data_path data --tta
  python test_dataset_pytorch.py --data_path data --auto 1500
  python test_dataset_pytorch.py --data_path data --save_errors

Controls:
  Any Key  - Next image
  A        - Enable auto mode (1.5s delay)
  M        - Manual mode (disable auto)
  S        - Save current display
  Q        - Quit
        """
    )
    
    parser.add_argument('--data_path', '-d', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file (.pth)')
    parser.add_argument('--samples', '-s', type=int, default=10,
                        help='Number of samples per class (default: 10)')
    parser.add_argument('--tta', action='store_true',
                        help='Enable Test-Time Augmentation')
    parser.add_argument('--auto', type=int, default=0,
                        help='Auto-advance delay in ms (0 = manual mode)')
    parser.add_argument('--save_errors', action='store_true',
                        help='Save incorrectly predicted images')
    parser.add_argument('--output_dir', '-o', type=str, default='output',
                        help='Output directory for saved images')
    
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
        results = test_on_dataset(
            model, args.data_path, args.samples, device,
            use_tta=args.tta,
            auto_delay=args.auto,
            save_errors=args.save_errors,
            output_dir=args.output_dir
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Print results
    print_results(results)
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
