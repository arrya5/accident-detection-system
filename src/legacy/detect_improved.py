"""
Accident Detection with TTA + Temporal Smoothing

Improvements over basic detection:
1. Test-Time Augmentation (5 predictions averaged)
2. Higher confidence threshold (0.6 instead of 0.5)
3. Temporal smoothing - requires 3+ consecutive accident frames to confirm
4. Reduces false positives significantly

Author: [Your Name]
Date: December 2025
"""

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
from collections import deque

# Configuration
IMG_SIZE = 224
TTA_AUGMENTATIONS = 5

# FALSE POSITIVE REDUCTION SETTINGS
CONFIDENCE_THRESHOLD = 0.65  # Higher threshold (was 0.5)
TEMPORAL_WINDOW = 5          # Look at last 5 frames
MIN_CONSECUTIVE = 3          # Need 3+ accident frames to confirm


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
    """Apply Test-Time Augmentations."""
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    
    augmented = []
    augmented.append(img)  # Original
    augmented.append(cv2.flip(img, 1))  # Flipped
    augmented.append(cv2.convertScaleAbs(img, alpha=1.1, beta=10))  # Bright
    augmented.append(cv2.convertScaleAbs(img, alpha=0.9, beta=-10))  # Dark
    augmented.append(cv2.convertScaleAbs(img, alpha=1.15, beta=0))  # Contrast
    
    return augmented


def predict_with_tta(model, frame):
    """Make prediction using Test-Time Augmentation."""
    augmented_frames = apply_tta_augmentations(frame)
    batch = np.array(augmented_frames)
    predictions = model.predict(batch, verbose=0)
    avg_prediction = np.mean(predictions)
    accident_prob = 1 - avg_prediction
    return accident_prob, predictions.flatten()


class TemporalSmoother:
    """
    Reduces false positives using temporal smoothing.
    
    Only confirms accident if:
    1. Current confidence > threshold
    2. At least MIN_CONSECUTIVE of last TEMPORAL_WINDOW frames were accidents
    """
    
    def __init__(self, window_size=5, min_consecutive=3, threshold=0.65):
        self.window = deque(maxlen=window_size)
        self.min_consecutive = min_consecutive
        self.threshold = threshold
    
    def update(self, confidence):
        """Update with new frame and return smoothed decision."""
        # Raw prediction based on threshold
        raw_accident = confidence >= self.threshold
        self.window.append(raw_accident)
        
        # Count recent accidents
        recent_accidents = sum(self.window)
        
        # Confirm only if enough consecutive accidents
        confirmed_accident = recent_accidents >= self.min_consecutive
        
        return confirmed_accident, raw_accident, recent_accidents


def create_overlay(frame, is_confirmed, is_raw, confidence, individual_preds,
                   recent_count, frame_num, total_frames, fps, stats):
    """Create visual overlay with temporal smoothing info."""
    display = frame.copy()
    h, w = display.shape[:2]
    
    # Border color based on CONFIRMED status (not raw)
    if is_confirmed:
        border_color = (0, 0, 255)  # Red - Confirmed accident
        status = "🚨 ACCIDENT CONFIRMED"
    elif is_raw:
        border_color = (0, 165, 255)  # Orange - Detected but not confirmed
        status = "⚠️ Possible Accident (awaiting confirmation)"
    else:
        border_color = (0, 255, 0)  # Green - Normal
        status = "✓ NORMAL TRAFFIC"
    
    cv2.rectangle(display, (0, 0), (w, h), border_color, 10)
    
    # Info panel
    panel_h = 200
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    
    # Header
    cv2.rectangle(panel, (0, 0), (w, 50), (50, 50, 50), -1)
    cv2.putText(panel, "TTA + TEMPORAL SMOOTHING (Reduces False Positives)",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    
    # Status
    cv2.putText(panel, status, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, border_color, 2)
    
    # Confidence bar
    bar_w = 300
    bar_x = 20
    bar_y = 95
    
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 22), (60, 60, 60), -1)
    fill = int(bar_w * confidence)
    bar_color = (0, 0, 255) if confidence >= CONFIDENCE_THRESHOLD else (0, 255, 0)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill, bar_y + 22), bar_color, -1)
    
    # Threshold line
    thresh_x = bar_x + int(bar_w * CONFIDENCE_THRESHOLD)
    cv2.line(panel, (thresh_x, bar_y - 5), (thresh_x, bar_y + 27), (255, 255, 0), 2)
    
    cv2.putText(panel, f"Confidence: {confidence*100:.1f}% (threshold: {CONFIDENCE_THRESHOLD*100:.0f}%)",
                (bar_x + bar_w + 15, bar_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Temporal window visualization
    cv2.putText(panel, f"Temporal Filter: {recent_count}/{TEMPORAL_WINDOW} recent frames are accidents",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(panel, f"(Need {MIN_CONSECUTIVE}+ to confirm)",
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    # Temporal window boxes
    box_start_x = 380
    for i in range(TEMPORAL_WINDOW):
        box_x = box_start_x + i * 25
        color = (60, 60, 60)  # Empty
        if i < recent_count:
            color = (0, 0, 200)  # Accident
        cv2.rectangle(panel, (box_x, 135), (box_x + 20, 155), color, -1)
        cv2.rectangle(panel, (box_x, 135), (box_x + 20, 155), (100, 100, 100), 1)
    
    # Stats
    progress = frame_num / total_frames * 100
    cv2.putText(panel, f"Frame: {frame_num}/{total_frames} ({progress:.1f}%)",
                (w - 280, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"Confirmed Accidents: {stats['confirmed']}",
                (w - 280, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(panel, f"False Positives Prevented: {stats['prevented']}",
                (w - 280, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Individual TTA predictions (smaller)
    aug_names = ["Orig", "Flip", "Brt", "Drk", "Ctr"]
    for i, (name, pred) in enumerate(zip(aug_names, individual_preds)):
        x = 20 + i * 80
        acc_prob = 1 - pred
        color = (0, 0, 200) if acc_prob >= 0.5 else (0, 200, 0)
        cv2.putText(panel, f"{name}:{acc_prob*100:.0f}%",
                    (x, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    # Combine
    display = np.vstack([panel, display])
    
    return display


def process_video(video_path, model, save_output=True):
    """Process video with TTA + Temporal Smoothing."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer for saving output
    video_writer = None
    output_path = None
    if save_output:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "detection_result.mp4")
        # Will initialize writer after first frame (to get correct dimensions with overlay)
    
    print(f"\n📹 Video: {video_path}")
    print(f"   Duration: {total_frames/fps:.1f}s ({total_frames} frames)")
    if save_output:
        print(f"\n💾 Output will be saved to: {output_path}")
    print(f"\n🛡️ False Positive Reduction Settings:")
    print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}% (was 50%)")
    print(f"   Temporal Window: {TEMPORAL_WINDOW} frames")
    print(f"   Min Consecutive: {MIN_CONSECUTIVE} frames to confirm")
    print("\n🔍 Processing... Press 'Q' to quit\n")
    
    # Initialize
    smoother = TemporalSmoother(TEMPORAL_WINDOW, MIN_CONSECUTIVE, CONFIDENCE_THRESHOLD)
    stats = {'confirmed': 0, 'raw': 0, 'prevented': 0}
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Predict with TTA
        confidence, individual_preds = predict_with_tta(model, frame)
        
        # Apply temporal smoothing
        is_confirmed, is_raw, recent_count = smoother.update(confidence)
        
        # Update stats
        if is_raw:
            stats['raw'] += 1
        if is_confirmed:
            stats['confirmed'] += 1
        if is_raw and not is_confirmed:
            stats['prevented'] += 1
        
        # Create display
        display = create_overlay(frame, is_confirmed, is_raw, confidence, 
                                individual_preds, recent_count, frame_count, 
                                total_frames, fps, stats)
        
        # Initialize video writer on first frame (now we know the dimensions)
        if save_output and video_writer is None:
            out_h, out_w = display.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
            print(f"   📝 Writing video at {out_w}x{out_h} @ {fps:.1f} FPS\n")
        
        # Write frame to output video
        if video_writer is not None:
            video_writer.write(display)
        
        # Resize for display only (not for saving)
        disp_h, disp_w = display.shape[:2]
        if disp_w > 1200:
            scale = 1200 / disp_w
            display = cv2.resize(display, (1200, int(disp_h * scale)))
        
        cv2.imshow('Accident Detection - False Positive Reduction', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 DETECTION SUMMARY (with False Positive Reduction)")
    print("=" * 70)
    print(f"   Total Frames: {frame_count}")
    print(f"   Raw Detections (before filtering): {stats['raw']} ({stats['raw']/frame_count*100:.1f}%)")
    print(f"   Confirmed Accidents (after filtering): {stats['confirmed']} ({stats['confirmed']/frame_count*100:.1f}%)")
    print(f"   🛡️ False Positives Prevented: {stats['prevented']}")
    if save_output and output_path:
        print(f"\n💾 Output video saved to: {output_path}")
    print("=" * 70)
    
    return stats, output_path


def main():
    print("=" * 70)
    print("🚀 ACCIDENT DETECTION - FALSE POSITIVE REDUCTION")
    print("   TTA + Higher Threshold + Temporal Smoothing")
    print("=" * 70)
    
    model = load_model()
    
    video_path = r"c:\Users\arrya\Downloads\test_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    stats, output_path = process_video(video_path, model, save_output=True)
    
    print("\n✅ Detection complete!")
    if output_path:
        print(f"📁 You can find the output video at:")
        print(f"   {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
