"""
Accident Detection System - Video Detection Module

This module provides real-time accident detection from video feeds
using a pre-trained MobileNetV2 deep learning model.

Author: [Your Name]
Date: December 2025
License: MIT
"""

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import argparse
import os
import sys

# Configuration
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5


def load_model(model_path: str = None) -> keras.Model:
    """
    Load the trained accident detection model.
    
    Args:
        model_path: Path to the model file. If None, searches default locations.
        
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If no model file is found
    """
    # Default model paths to search
    default_paths = [
        "models/accident_detector.keras",
        "models/accident_detector.h5",
        "../models/accident_detector.keras",
        "../models/accident_detector.h5",
    ]
    
    if model_path:
        paths_to_try = [model_path]
    else:
        paths_to_try = default_paths
    
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"✅ Loading model from: {path}")
            return keras.models.load_model(path)
    
    raise FileNotFoundError(
        f"Model not found. Searched: {paths_to_try}\n"
        "Please ensure the model file exists or provide --model_path argument."
    )


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a video frame for model input.
    
    Args:
        frame: BGR image from OpenCV (H, W, 3)
        
    Returns:
        Preprocessed image array (1, 224, 224, 3)
    """
    # Resize to model input size
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    
    # Add batch dimension
    img_array = np.expand_dims(img, axis=0)
    
    return img_array


def predict_accident(model: keras.Model, frame: np.ndarray) -> tuple:
    """
    Predict whether a frame contains an accident.
    
    Args:
        model: Trained Keras model
        frame: Preprocessed image array
        
    Returns:
        Tuple of (is_accident: bool, accident_probability: float)
    """
    # Get prediction
    prediction = model.predict(frame, verbose=0)[0][0]
    
    # Model outputs P(Normal), so P(Accident) = 1 - P(Normal)
    accident_prob = 1 - prediction
    
    is_accident = accident_prob >= CONFIDENCE_THRESHOLD
    
    return is_accident, accident_prob


def create_overlay(frame: np.ndarray, is_accident: bool, confidence: float,
                   frame_num: int, total_frames: int, fps: float) -> np.ndarray:
    """
    Create visual overlay with detection results.
    
    Args:
        frame: Original video frame
        is_accident: Whether accident was detected
        confidence: Confidence score (0-1)
        frame_num: Current frame number
        total_frames: Total frames in video
        fps: Video FPS
        
    Returns:
        Frame with overlay
    """
    display = frame.copy()
    h, w = display.shape[:2]
    
    # Border color based on detection
    if is_accident:
        border_color = (0, 0, 255)  # Red for accident
        status_text = "⚠️ ACCIDENT DETECTED"
        status_color = (0, 0, 255)
    else:
        border_color = (0, 255, 0)  # Green for normal
        status_text = "✓ NORMAL TRAFFIC"
        status_color = (0, 255, 0)
    
    # Draw border
    cv2.rectangle(display, (0, 0), (w, h), border_color, 8)
    
    # Create info panel
    panel_height = 120
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background
    
    # Status text
    cv2.putText(panel, status_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    
    # Confidence bar
    bar_width = 300
    bar_x = 20
    bar_y = 55
    bar_height = 20
    
    # Background bar
    cv2.rectangle(panel, (bar_x, bar_y), 
                  (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
    
    # Confidence fill
    fill_width = int(bar_width * confidence)
    fill_color = (0, 0, 255) if is_accident else (0, 255, 0)
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + fill_width, bar_y + bar_height), fill_color, -1)
    
    # Confidence text
    cv2.putText(panel, f"Confidence: {confidence*100:.1f}%",
                (bar_x + bar_width + 20, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Progress info
    current_time = frame_num / fps
    total_time = total_frames / fps
    progress = frame_num / total_frames * 100
    
    cv2.putText(panel, f"Frame: {frame_num}/{total_frames} ({progress:.1f}%)",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"Time: {current_time:.1f}s / {total_time:.1f}s",
                (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Combine panel and frame
    display = np.vstack([panel, display])
    
    return display


def process_video(video_path: str, model: keras.Model, 
                  output_path: str = None, show_display: bool = True) -> dict:
    """
    Process a video file and detect accidents.
    
    Args:
        video_path: Path to input video
        model: Trained Keras model
        output_path: Path to save output video (optional)
        show_display: Whether to show real-time display
        
    Returns:
        Dictionary with detection statistics
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video: {video_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Duration: {total_frames/fps:.1f} seconds ({total_frames} frames)")
    print(f"   FPS: {fps:.1f}")
    
    # Initialize video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 120))
    
    # Statistics
    accident_frames = 0
    frame_count = 0
    confidences = []
    
    print("\n🔍 Processing video... Press 'Q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Preprocess and predict
        processed = preprocess_frame(frame)
        is_accident, confidence = predict_accident(model, processed)
        
        if is_accident:
            accident_frames += 1
        confidences.append(confidence)
        
        # Create overlay
        display = create_overlay(frame, is_accident, confidence,
                                frame_count, total_frames, fps)
        
        # Write to output video
        if writer:
            writer.write(display)
        
        # Show display
        if show_display:
            # Resize if too large
            display_h, display_w = display.shape[:2]
            if display_w > 1200:
                scale = 1200 / display_w
                display = cv2.resize(display, (1200, int(display_h * scale)))
            
            cv2.imshow('Accident Detection System', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n⏹️ Detection stopped by user")
                break
            elif key == ord('p') or key == ord('P'):
                print("⏸️ Paused. Press any key to continue...")
                cv2.waitKey(0)
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Calculate statistics
    stats = {
        'total_frames': frame_count,
        'accident_frames': accident_frames,
        'accident_percentage': accident_frames / frame_count * 100 if frame_count > 0 else 0,
        'avg_confidence': np.mean(confidences) if confidences else 0,
        'max_confidence': max(confidences) if confidences else 0,
        'min_confidence': min(confidences) if confidences else 0,
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DETECTION SUMMARY")
    print("=" * 60)
    print(f"   Total Frames Analyzed: {stats['total_frames']}")
    print(f"   Accident Frames: {stats['accident_frames']} ({stats['accident_percentage']:.1f}%)")
    print(f"   Normal Frames: {stats['total_frames'] - stats['accident_frames']}")
    print(f"   Average Confidence: {stats['avg_confidence']*100:.1f}%")
    print(f"   Confidence Range: {stats['min_confidence']*100:.1f}% - {stats['max_confidence']*100:.1f}%")
    print("=" * 60)
    
    if output_path:
        print(f"\n💾 Output saved to: {output_path}")
    
    return stats


def main():
    """Main entry point for video detection."""
    parser = argparse.ArgumentParser(
        description='Real-time Accident Detection from Video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect.py --video traffic.mp4
  python detect.py --video traffic.mp4 --output result.mp4
  python detect.py --video traffic.mp4 --model_path models/custom_model.keras
        """
    )
    
    parser.add_argument('--video', '-v', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save output video (optional)')
    parser.add_argument('--model_path', '-m', type=str, default=None,
                        help='Path to model file (optional)')
    parser.add_argument('--no_display', action='store_true',
                        help='Disable real-time display (faster processing)')
    
    args = parser.parse_args()
    
    # Check video exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video not found: {args.video}")
        sys.exit(1)
    
    print("=" * 60)
    print("🚗 REAL-TIME ACCIDENT DETECTION SYSTEM")
    print("   Using MobileNetV2 Deep Learning Model")
    print("=" * 60)
    
    # Load model
    try:
        model = load_model(args.model_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Process video
    try:
        stats = process_video(
            args.video, 
            model, 
            output_path=args.output,
            show_display=not args.no_display
        )
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        sys.exit(1)
    
    print("\n✅ Detection complete!")


if __name__ == "__main__":
    main()
