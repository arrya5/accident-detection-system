"""
Accident Detection with Test-Time Augmentation (TTA)

Test-Time Augmentation improves accuracy by:
1. Creating multiple augmented versions of each frame
2. Getting predictions for all versions
3. Averaging predictions for more robust results

Expected improvement: +2-3% accuracy

Author: [Your Name]
Date: December 2025
"""

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os

# Configuration
IMG_SIZE = 224
TTA_AUGMENTATIONS = 5  # Number of augmented versions per frame


def load_model():
    """Load the trained model."""
    paths = [
        "models/accident_detector.keras",
        "models/accident_detector.h5",
        "../models/accident_detector.keras",
        "../models/accident_detector.h5",
    ]
    for path in paths:
        if os.path.exists(path):
            print(f"✅ Model loaded: {path}")
            return keras.models.load_model(path)
    raise FileNotFoundError("Model not found!")


def apply_tta_augmentations(frame):
    """
    Apply Test-Time Augmentations to a single frame.
    
    Returns list of augmented frames for ensemble prediction.
    """
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    
    augmented = []
    
    # 1. Original
    augmented.append(img)
    
    # 2. Horizontal flip
    augmented.append(cv2.flip(img, 1))
    
    # 3. Slight brightness increase
    bright = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    augmented.append(bright)
    
    # 4. Slight brightness decrease
    dark = cv2.convertScaleAbs(img, alpha=0.9, beta=-10)
    augmented.append(dark)
    
    # 5. Slight contrast adjustment
    contrast = cv2.convertScaleAbs(img, alpha=1.15, beta=0)
    augmented.append(contrast)
    
    return augmented


def predict_with_tta(model, frame):
    """
    Make prediction using Test-Time Augmentation.
    
    Returns averaged prediction from multiple augmented versions.
    """
    # Get augmented versions
    augmented_frames = apply_tta_augmentations(frame)
    
    # Stack into batch
    batch = np.array(augmented_frames)
    
    # Get predictions for all augmented versions
    predictions = model.predict(batch, verbose=0)
    
    # Average predictions
    avg_prediction = np.mean(predictions)
    
    # Convert to accident probability
    accident_prob = 1 - avg_prediction
    
    return accident_prob, predictions.flatten()


def create_tta_overlay(frame, is_accident, confidence, individual_preds,
                       frame_num, total_frames, fps, stats):
    """Create visual overlay showing TTA predictions."""
    display = frame.copy()
    h, w = display.shape[:2]
    
    # Border color
    if is_accident:
        border_color = (0, 0, 255)  # Red
        status = "⚠️ ACCIDENT DETECTED"
    else:
        border_color = (0, 255, 0)  # Green
        status = "✓ NORMAL TRAFFIC"
    
    cv2.rectangle(display, (0, 0), (w, h), border_color, 10)
    
    # Create info panel
    panel_h = 180
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    
    # Header
    cv2.rectangle(panel, (0, 0), (w, 50), (50, 50, 50), -1)
    cv2.putText(panel, "TEST-TIME AUGMENTATION (TTA) - 5 Predictions Averaged",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Status
    cv2.putText(panel, status, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 2)
    
    # Main confidence bar
    bar_w = 300
    bar_x = 20
    bar_y = 95
    
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 25), (60, 60, 60), -1)
    fill = int(bar_w * confidence)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill, bar_y + 25), border_color, -1)
    cv2.putText(panel, f"TTA Confidence: {confidence*100:.1f}%",
                (bar_x + bar_w + 15, bar_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Individual TTA predictions (small bars)
    aug_names = ["Original", "Flipped", "Bright", "Dark", "Contrast"]
    start_y = 130
    small_bar_w = 80
    
    for i, (name, pred) in enumerate(zip(aug_names, individual_preds)):
        x = 20 + i * 150
        acc_prob = 1 - pred
        
        # Mini bar
        cv2.rectangle(panel, (x, start_y), (x + small_bar_w, start_y + 12), (60, 60, 60), -1)
        fill = int(small_bar_w * acc_prob)
        color = (0, 0, 255) if acc_prob >= 0.5 else (0, 255, 0)
        cv2.rectangle(panel, (x, start_y), (x + fill, start_y + 12), color, -1)
        
        # Label
        cv2.putText(panel, f"{name}: {acc_prob*100:.0f}%",
                    (x, start_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    # Progress and stats
    progress = frame_num / total_frames * 100
    cv2.putText(panel, f"Frame: {frame_num}/{total_frames} ({progress:.1f}%)",
                (w - 280, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"Accidents: {stats['accidents']}/{frame_num} ({stats['acc_pct']:.1f}%)",
                (w - 280, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Combine
    display = np.vstack([panel, display])
    
    return display


def process_video_with_tta(video_path, model):
    """Process video with Test-Time Augmentation."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📹 Video: {video_path}")
    print(f"   Duration: {total_frames/fps:.1f}s ({total_frames} frames)")
    print(f"   Using {TTA_AUGMENTATIONS} augmentations per frame")
    print("\n🔍 Processing with TTA... Press 'Q' to quit\n")
    
    # Statistics
    stats = {'accidents': 0, 'acc_pct': 0}
    frame_count = 0
    confidences = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Predict with TTA
        confidence, individual_preds = predict_with_tta(model, frame)
        is_accident = confidence >= 0.5
        
        if is_accident:
            stats['accidents'] += 1
        stats['acc_pct'] = stats['accidents'] / frame_count * 100
        confidences.append(confidence)
        
        # Create display
        display = create_tta_overlay(frame, is_accident, confidence, individual_preds,
                                     frame_count, total_frames, fps, stats)
        
        # Resize for display
        disp_h, disp_w = display.shape[:2]
        if disp_w > 1200:
            scale = 1200 / disp_w
            display = cv2.resize(display, (1200, int(disp_h * scale)))
        
        cv2.imshow('Accident Detection with TTA', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TTA DETECTION SUMMARY")
    print("=" * 70)
    print(f"   Total Frames: {frame_count}")
    print(f"   Accident Frames: {stats['accidents']} ({stats['acc_pct']:.1f}%)")
    print(f"   Normal Frames: {frame_count - stats['accidents']}")
    print(f"   Avg Confidence: {np.mean(confidences)*100:.1f}%")
    print(f"   Confidence Range: {min(confidences)*100:.1f}% - {max(confidences)*100:.1f}%")
    print("=" * 70)
    
    return stats


def main():
    print("=" * 70)
    print("🚀 ACCIDENT DETECTION WITH TEST-TIME AUGMENTATION (TTA)")
    print("   5 augmented predictions averaged per frame for +2-3% accuracy")
    print("=" * 70)
    
    # Load model
    model = load_model()
    
    # Video path
    video_path = r"c:\Users\arrya\Downloads\test_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Process with TTA
    stats = process_video_with_tta(video_path, model)
    
    print("\n✅ TTA Detection complete!")


if __name__ == "__main__":
    main()
