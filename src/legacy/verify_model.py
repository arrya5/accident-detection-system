"""
Accident Detection System - Model Verification Module

This module verifies that the trained model is genuinely learning patterns
and not memorizing the training data (anti-overfitting/anti-cheating check).

Author: [Your Name]
Date: December 2025
License: MIT

Verification Methodology:
    1. Test on COMPLETELY UNSEEN data (validation/test set)
    2. Analyze confidence distribution (should be wide, not binary)
    3. Check accuracy is in healthy range (70-90%, not 99%+)
    4. Verify model works on both classes
"""

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import argparse
import sys

# Configuration
IMG_SIZE = 224


def load_model(model_path: str = None) -> keras.Model:
    """Load the trained model."""
    default_paths = [
        "models/accident_detector.keras",
        "models/accident_detector.h5",
        "../models/accident_detector.keras",
        "../models/accident_detector.h5",
    ]
    
    paths_to_try = [model_path] if model_path else default_paths
    
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"✅ Loading model from: {path}")
            return keras.models.load_model(path)
    
    raise FileNotFoundError(f"Model not found. Searched: {paths_to_try}")


def verify_on_dataset(model: keras.Model, data_path: str, 
                      visual: bool = False) -> dict:
    """
    Verify model on unseen test/validation data.
    
    Args:
        model: Trained Keras model
        data_path: Path to dataset root
        visual: Whether to show visual verification
        
    Returns:
        Dictionary with verification results
    """
    # Find test directory
    test_dir = os.path.join(data_path, "val")  # Use validation set
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_path, "test")
    
    accident_dir = os.path.join(test_dir, "Accident")
    normal_dir = os.path.join(test_dir, "Non Accident")
    
    if not os.path.exists(accident_dir) or not os.path.exists(normal_dir):
        raise FileNotFoundError(f"Test directories not found in: {test_dir}")
    
    # Get all images
    accident_images = [f for f in os.listdir(accident_dir) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    normal_images = [f for f in os.listdir(normal_dir) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"\n📂 Testing on UNSEEN data:")
    print(f"   Accident images: {len(accident_images)}")
    print(f"   Normal images: {len(normal_images)}")
    print(f"   Total: {len(accident_images) + len(normal_images)}")
    print("\n   ⚠️  This data was NEVER used during training!\n")
    
    # Test accident images
    accident_correct = 0
    accident_confidences = []
    
    print("🔍 Testing accident images...")
    for img_name in accident_images:
        img_path = os.path.join(accident_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)
        
        prediction = model.predict(img_array, verbose=0)[0][0]
        accident_prob = 1 - prediction
        
        if accident_prob >= 0.5:
            accident_correct += 1
        accident_confidences.append(accident_prob)
        
        if visual:
            show_prediction(img, "ACCIDENT", accident_prob >= 0.5, accident_prob)
    
    # Test normal images
    normal_correct = 0
    normal_confidences = []
    
    print("🔍 Testing normal traffic images...")
    for img_name in normal_images:
        img_path = os.path.join(normal_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)
        
        prediction = model.predict(img_array, verbose=0)[0][0]
        normal_prob = prediction
        
        if normal_prob >= 0.5:
            normal_correct += 1
        normal_confidences.append(normal_prob)
        
        if visual:
            show_prediction(img, "NORMAL", normal_prob >= 0.5, normal_prob)
    
    # Calculate statistics
    total_tested = len(accident_images) + len(normal_images)
    total_correct = accident_correct + normal_correct
    overall_accuracy = total_correct / total_tested * 100 if total_tested > 0 else 0
    
    accident_accuracy = accident_correct / len(accident_images) * 100 if accident_images else 0
    normal_accuracy = normal_correct / len(normal_images) * 100 if normal_images else 0
    
    results = {
        'total_tested': total_tested,
        'total_correct': total_correct,
        'overall_accuracy': overall_accuracy,
        'accident_tested': len(accident_images),
        'accident_correct': accident_correct,
        'accident_accuracy': accident_accuracy,
        'accident_avg_confidence': np.mean(accident_confidences) * 100 if accident_confidences else 0,
        'accident_min_confidence': min(accident_confidences) * 100 if accident_confidences else 0,
        'accident_max_confidence': max(accident_confidences) * 100 if accident_confidences else 0,
        'normal_tested': len(normal_images),
        'normal_correct': normal_correct,
        'normal_accuracy': normal_accuracy,
        'normal_avg_confidence': np.mean(normal_confidences) * 100 if normal_confidences else 0,
        'normal_min_confidence': min(normal_confidences) * 100 if normal_confidences else 0,
        'normal_max_confidence': max(normal_confidences) * 100 if normal_confidences else 0,
    }
    
    return results


def show_prediction(img: np.ndarray, true_label: str, 
                    is_correct: bool, confidence: float):
    """Show visual prediction (for visual mode)."""
    display = img.copy()
    h, w = display.shape[:2]
    
    # Resize if too large
    if w > 800:
        scale = 800 / w
        display = cv2.resize(display, (800, int(h * scale)))
        h, w = display.shape[:2]
    
    # Border color
    color = (0, 255, 0) if is_correct else (0, 0, 255)
    cv2.rectangle(display, (0, 0), (w, h), color, 8)
    
    # Text
    result = "CORRECT" if is_correct else "WRONG"
    cv2.putText(display, f"True: {true_label} | {result}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display, f"Confidence: {confidence*100:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Model Verification', display)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        sys.exit(0)


def print_verification_report(results: dict):
    """Print detailed verification report."""
    print("\n" + "=" * 70)
    print("📊 MODEL VERIFICATION REPORT")
    print("=" * 70)
    
    print(f"\n📈 OVERALL PERFORMANCE")
    print(f"   Total Images Tested: {results['total_tested']}")
    print(f"   Correctly Classified: {results['total_correct']}")
    print(f"   Overall Accuracy: {results['overall_accuracy']:.1f}%")
    
    print(f"\n🔴 ACCIDENT DETECTION")
    print(f"   Images Tested: {results['accident_tested']}")
    print(f"   Correctly Detected: {results['accident_correct']}")
    print(f"   Accuracy: {results['accident_accuracy']:.1f}%")
    print(f"   Avg Confidence: {results['accident_avg_confidence']:.1f}%")
    print(f"   Confidence Range: {results['accident_min_confidence']:.1f}% - {results['accident_max_confidence']:.1f}%")
    
    print(f"\n🟢 NORMAL TRAFFIC DETECTION")
    print(f"   Images Tested: {results['normal_tested']}")
    print(f"   Correctly Detected: {results['normal_correct']}")
    print(f"   Accuracy: {results['normal_accuracy']:.1f}%")
    print(f"   Avg Confidence: {results['normal_avg_confidence']:.1f}%")
    print(f"   Confidence Range: {results['normal_min_confidence']:.1f}% - {results['normal_max_confidence']:.1f}%")
    
    # Verification verdict
    print("\n" + "=" * 70)
    print("🔍 VERIFICATION VERDICT")
    print("=" * 70)
    
    # Check for overfitting
    accuracy = results['overall_accuracy']
    confidence_range = max(
        results['accident_max_confidence'] - results['accident_min_confidence'],
        results['normal_max_confidence'] - results['normal_min_confidence']
    )
    
    issues = []
    passes = []
    
    # Accuracy check
    if accuracy >= 95:
        issues.append("⚠️  Accuracy too high (>95%) - possible overfitting")
    elif accuracy < 60:
        issues.append("⚠️  Accuracy too low (<60%) - model not learning")
    else:
        passes.append("✅ Accuracy in healthy range (60-95%)")
    
    # Confidence range check
    if confidence_range < 20:
        issues.append("⚠️  Confidence range too narrow - possible memorization")
    else:
        passes.append("✅ Wide confidence range - genuine analysis")
    
    # Class balance check
    acc_diff = abs(results['accident_accuracy'] - results['normal_accuracy'])
    if acc_diff > 30:
        issues.append(f"⚠️  Large accuracy gap between classes ({acc_diff:.1f}%)")
    else:
        passes.append("✅ Balanced performance on both classes")
    
    # Print results
    for p in passes:
        print(f"   {p}")
    for i in issues:
        print(f"   {i}")
    
    print()
    if not issues:
        print("   🎉 MODEL PASSED VERIFICATION!")
        print("   The model is learning genuine patterns, not memorizing data.")
    else:
        print("   ⚠️  MODEL HAS POTENTIAL ISSUES")
        print("   Review the issues above and consider retraining.")
    
    print("=" * 70)


def main():
    """Main entry point for model verification."""
    parser = argparse.ArgumentParser(
        description='Verify Accident Detection Model (Anti-Cheat Check)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool verifies that the model is genuinely learning patterns
and not just memorizing the training data.

Examples:
  python verify_model.py --data_path /path/to/dataset
  python verify_model.py --data_path /path/to/dataset --visual
        """
    )
    
    parser.add_argument('--data_path', '-d', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--model_path', '-m', type=str, default=None,
                        help='Path to model file')
    parser.add_argument('--visual', '-v', action='store_true',
                        help='Show visual verification (displays each image)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 MODEL VERIFICATION (ANTI-CHEAT CHECK)")
    print("   Testing on completely UNSEEN data")
    print("=" * 70)
    
    # Load model
    try:
        model = load_model(args.model_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Run verification
    try:
        results = verify_on_dataset(model, args.data_path, visual=args.visual)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    if args.visual:
        cv2.destroyAllWindows()
    
    # Print report
    print_verification_report(results)
    
    print("\n✅ Verification complete!")


if __name__ == "__main__":
    main()
