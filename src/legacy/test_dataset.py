"""
Accident Detection System - Dataset Testing Module

This module allows testing the trained model on random samples
from the test dataset with visual feedback.

Author: [Your Name]
Date: December 2025
License: MIT
"""

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import random
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


def test_on_dataset(model: keras.Model, data_path: str, 
                    num_samples: int = 10) -> dict:
    """
    Test model on random samples from the dataset.
    
    Args:
        model: Trained Keras model
        data_path: Path to dataset root
        num_samples: Number of samples to test per class
        
    Returns:
        Dictionary with test results
    """
    # Find test directory
    test_dir = os.path.join(data_path, "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_path, "val")
    
    accident_dir = os.path.join(test_dir, "Accident")
    normal_dir = os.path.join(test_dir, "Non Accident")
    
    if not os.path.exists(accident_dir) or not os.path.exists(normal_dir):
        raise FileNotFoundError(f"Test directories not found in: {test_dir}")
    
    # Get random samples
    accident_images = [f for f in os.listdir(accident_dir) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    normal_images = [f for f in os.listdir(normal_dir) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    sample_accidents = random.sample(accident_images, 
                                      min(num_samples, len(accident_images)))
    sample_normals = random.sample(normal_images, 
                                    min(num_samples, len(normal_images)))
    
    print(f"\n📂 Testing on {len(sample_accidents)} accident + {len(sample_normals)} normal images")
    print("   Press any key for next image, 'Q' to quit\n")
    
    # Statistics
    accident_correct = 0
    normal_correct = 0
    
    # Test accident images
    for idx, img_name in enumerate(sample_accidents, 1):
        img_path = os.path.join(accident_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Predict
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_array, verbose=0)[0][0]
        accident_prob = 1 - prediction
        
        is_correct = accident_prob >= 0.5
        if is_correct:
            accident_correct += 1
        
        # Display
        display = create_display(img, "ACCIDENT", is_correct, 
                                 accident_prob, idx, len(sample_accidents))
        
        cv2.imshow('Dataset Testing', display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    # Test normal images
    for idx, img_name in enumerate(sample_normals, 1):
        img_path = os.path.join(normal_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Predict
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_array, verbose=0)[0][0]
        normal_prob = prediction
        
        is_correct = normal_prob >= 0.5
        if is_correct:
            normal_correct += 1
        
        # Display
        display = create_display(img, "NORMAL", is_correct, 
                                 normal_prob, idx, len(sample_normals))
        
        cv2.imshow('Dataset Testing', display)
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


def create_display(img: np.ndarray, true_label: str, is_correct: bool,
                   confidence: float, idx: int, total: int) -> np.ndarray:
    """Create display with prediction overlay."""
    display = img.copy()
    h, w = display.shape[:2]
    
    # Resize if needed
    if w > 800:
        scale = 800 / w
        display = cv2.resize(display, (800, int(h * scale)))
        h, w = display.shape[:2]
    
    # Border color
    color = (0, 255, 0) if is_correct else (0, 0, 255)
    cv2.rectangle(display, (0, 0), (w, h), color, 10)
    
    # Create info panel
    panel_height = 120
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    
    # True label
    cv2.putText(panel, f"True Label: {true_label} ({idx}/{total})", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Prediction result
    result = "✓ CORRECT" if is_correct else "✗ WRONG"
    cv2.putText(panel, f"Prediction: {result}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Confidence bar
    bar_w = 300
    bar_x = 20
    bar_y = 80
    bar_h = 20
    
    cv2.rectangle(panel, (bar_x, bar_y), 
                  (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * confidence)
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + bar_h), color, -1)
    
    cv2.putText(panel, f"Confidence: {confidence*100:.1f}%",
                (bar_x + bar_w + 20, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Combine
    display = np.vstack([panel, display])
    
    return display


def print_results(results: dict):
    """Print test results summary."""
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"   Total Tested: {results['total_tested']}")
    print(f"   Correct: {results['total_correct']}")
    print(f"   Accuracy: {results['accuracy']:.1f}%")
    print(f"\n   Accident: {results['accident_correct']}/{results['accident_total']}")
    print(f"   Normal: {results['normal_correct']}/{results['normal_total']}")
    print("=" * 50)


def main():
    """Main entry point for dataset testing."""
    parser = argparse.ArgumentParser(
        description='Test Model on Dataset Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_dataset.py --data_path /path/to/dataset
  python test_dataset.py --data_path /path/to/dataset --samples 20
        """
    )
    
    parser.add_argument('--data_path', '-d', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--model_path', '-m', type=str, default=None,
                        help='Path to model file')
    parser.add_argument('--samples', '-s', type=int, default=10,
                        help='Number of samples per class')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🧪 DATASET TESTING")
    print("   Testing on random samples")
    print("=" * 50)
    
    # Load model
    try:
        model = load_model(args.model_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Run tests
    try:
        results = test_on_dataset(model, args.data_path, args.samples)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Print results
    print_results(results)
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
