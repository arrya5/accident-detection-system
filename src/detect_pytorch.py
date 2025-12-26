"""
Accident Detection System - PyTorch Video Detection Module

Real-time accident detection from video feeds using a trained
MobileNetV2 PyTorch model with GPU acceleration.

Features:
- Real-time video processing with GPU acceleration
- Test-Time Augmentation (TTA) for improved accuracy
- Temporal smoothing to reduce false positives
- Support for webcam, video files, and RTSP streams
- Visual overlay with detection status and statistics

Author: Accident Detection System
Date: December 2025
License: MIT
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import argparse
import os
import sys
from collections import deque
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6  # Threshold for accident detection
TTA_ENABLED = True          # Enable Test-Time Augmentation
TEMPORAL_WINDOW = 5         # Number of frames for temporal smoothing
MIN_CONSECUTIVE = 3         # Minimum consecutive frames to confirm accident

# Colors (BGR format for OpenCV)
COLOR_SAFE = (0, 255, 0)      # Green
COLOR_DANGER = (0, 0, 255)    # Red
COLOR_WARNING = (0, 165, 255) # Orange
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


# ============================================================================
# MODEL DEFINITION (must match training)
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


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(model_path=None, device=None):
    """
    Load the trained PyTorch model.
    
    Args:
        model_path: Path to the .pth model file
        device: torch device (cuda or cpu)
        
    Returns:
        Loaded model and device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Default model paths to search
    default_paths = [
        "models/accident_detector_best.pth",
        "models/accident_detector.pth",
        "../models/accident_detector_best.pth",
        "../models/accident_detector.pth",
    ]
    
    paths_to_try = [model_path] if model_path else default_paths
    
    for path in paths_to_try:
        if path and os.path.exists(path):
            print(f"✅ Loading model from: {path}")
            
            # Create model
            model = AccidentDetector(num_classes=1, pretrained=False)
            
            # Load weights
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
            
            print(f"   Device: {device}")
            if device.type == 'cuda':
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
            
            return model, device
    
    raise FileNotFoundError(
        f"Model not found. Searched: {paths_to_try}\n"
        "Please ensure the model file exists or provide --model_path argument."
    )


# ============================================================================
# PREPROCESSING
# ============================================================================
def get_transforms():
    """Get image transforms for inference."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_frame(frame, transform):
    """
    Preprocess a video frame for model input.
    
    Args:
        frame: BGR image from OpenCV (H, W, 3)
        transform: torchvision transforms
        
    Returns:
        Preprocessed tensor (1, 3, 224, 224)
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    tensor = transform(rgb_frame)
    
    # Add batch dimension
    return tensor.unsqueeze(0)


# ============================================================================
# TTA (TEST-TIME AUGMENTATION)
# ============================================================================
def apply_tta(frame, transform):
    """
    Apply Test-Time Augmentation for more robust predictions.
    
    Args:
        frame: BGR image from OpenCV
        transform: torchvision transforms
        
    Returns:
        Batch of augmented tensors (N, 3, 224, 224)
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    augmented = []
    
    # Original
    augmented.append(transform(rgb_frame))
    
    # Horizontal flip
    flipped = cv2.flip(rgb_frame, 1)
    augmented.append(transform(flipped))
    
    # Brightness variations
    bright = cv2.convertScaleAbs(rgb_frame, alpha=1.1, beta=15)
    augmented.append(transform(bright))
    
    dark = cv2.convertScaleAbs(rgb_frame, alpha=0.9, beta=-15)
    augmented.append(transform(dark))
    
    # Contrast
    contrast = cv2.convertScaleAbs(rgb_frame, alpha=1.2, beta=0)
    augmented.append(transform(contrast))
    
    return torch.stack(augmented)


# ============================================================================
# TEMPORAL SMOOTHING
# ============================================================================
class TemporalSmoother:
    """
    Reduces false positives using temporal smoothing.
    
    Only confirms accident if multiple consecutive frames show accident.
    """
    
    def __init__(self, window_size=5, min_consecutive=3, threshold=0.6):
        self.window = deque(maxlen=window_size)
        self.min_consecutive = min_consecutive
        self.threshold = threshold
        self.confidence_history = deque(maxlen=window_size)
    
    def update(self, confidence):
        """
        Update with new frame and return smoothed decision.
        
        Args:
            confidence: Accident probability (0-1)
            
        Returns:
            tuple: (confirmed_accident, raw_prediction, recent_count)
        """
        raw_accident = confidence >= self.threshold
        self.window.append(raw_accident)
        self.confidence_history.append(confidence)
        
        recent_accidents = sum(self.window)
        confirmed_accident = recent_accidents >= self.min_consecutive
        
        return confirmed_accident, raw_accident, recent_accidents
    
    def get_avg_confidence(self):
        """Get average confidence over the window."""
        if len(self.confidence_history) == 0:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)


# ============================================================================
# PREDICTION
# ============================================================================
@torch.no_grad()
def predict(model, frame, device, transform, use_tta=True):
    """
    Predict accident probability for a frame.
    
    Args:
        model: Trained PyTorch model
        frame: BGR image from OpenCV
        device: torch device
        transform: torchvision transforms
        use_tta: Whether to use Test-Time Augmentation
        
    Returns:
        tuple: (accident_probability, individual_predictions)
        
    Note:
        Model outputs P(Non-Accident), so P(Accident) = 1 - sigmoid(output)
        This is because classes are: 0=Accident, 1=Non-Accident
    """
    if use_tta:
        # Apply TTA
        batch = apply_tta(frame, transform).to(device)
        outputs = model(batch)
        probabilities = torch.sigmoid(outputs).squeeze()
        
        # Model predicts P(Non-Accident), so P(Accident) = 1 - P(Non-Accident)
        accident_probs = 1 - probabilities
        
        # Average predictions
        avg_prob = accident_probs.mean().item()
        individual = accident_probs.cpu().numpy()
    else:
        # Single prediction
        tensor = preprocess_frame(frame, transform).to(device)
        output = model(tensor)
        non_accident_prob = torch.sigmoid(output).item()
        avg_prob = 1 - non_accident_prob  # P(Accident) = 1 - P(Non-Accident)
        individual = np.array([avg_prob])
    
    return avg_prob, individual


# ============================================================================
# VISUALIZATION
# ============================================================================
def draw_overlay(frame, is_confirmed, is_raw, confidence, avg_confidence,
                 recent_count, frame_num, total_frames, fps, stats):
    """
    Draw detection overlay on frame.
    
    Args:
        frame: OpenCV BGR image
        is_confirmed: Whether accident is confirmed (after temporal smoothing)
        is_raw: Raw prediction for this frame
        confidence: Current frame confidence
        avg_confidence: Average confidence over temporal window
        recent_count: Number of recent accident frames
        frame_num: Current frame number
        total_frames: Total frames (0 for live feed)
        fps: Current FPS
        stats: Detection statistics dictionary
        
    Returns:
        Frame with overlay
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Status bar at top
    status_height = 80
    if is_confirmed:
        status_color = COLOR_DANGER
        status_text = "🚨 ACCIDENT DETECTED"
    elif is_raw:
        status_color = COLOR_WARNING
        status_text = "⚠️ POSSIBLE ACCIDENT"
    else:
        status_color = COLOR_SAFE
        status_text = "✅ NORMAL TRAFFIC"
    
    # Draw status bar
    cv2.rectangle(overlay, (0, 0), (w, status_height), status_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Status text
    cv2.putText(frame, status_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3)
    
    # Confidence bar
    bar_width = 300
    bar_height = 25
    bar_x = w - bar_width - 20
    bar_y = 100
    
    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  COLOR_WHITE, -1)
    
    # Fill based on confidence
    fill_width = int(bar_width * confidence)
    fill_color = COLOR_DANGER if confidence > 0.6 else COLOR_WARNING if confidence > 0.4 else COLOR_SAFE
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                  fill_color, -1)
    
    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  COLOR_BLACK, 2)
    
    # Confidence text
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", 
                (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    # Statistics panel
    panel_y = 150
    line_height = 25
    
    info_lines = [
        f"Frame: {frame_num}" + (f"/{total_frames}" if total_frames > 0 else ""),
        f"FPS: {fps:.1f}",
        f"Temporal: {recent_count}/{TEMPORAL_WINDOW} frames",
        f"Avg Conf: {avg_confidence*100:.1f}%",
        f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%",
        "",
        f"Accidents: {stats['accidents']}",
        f"Total Frames: {stats['total_frames']}",
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (20, panel_y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    # Instructions
    cv2.putText(frame, "Press 'Q' to quit | 'S' to screenshot | 'T' toggle TTA",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    return frame


# ============================================================================
# MAIN DETECTION LOOP
# ============================================================================
def detect_video(source, model_path=None, output_path=None, show_display=True):
    """
    Run accident detection on a video source.
    
    Args:
        source: Video file path, webcam index (0, 1), or RTSP URL
        model_path: Path to .pth model file
        output_path: Path to save output video (optional)
        show_display: Whether to show live display
    """
    print("\n" + "="*60)
    print("🚗 ACCIDENT DETECTION SYSTEM (PyTorch)")
    print("="*60)
    
    # Load model
    model, device = load_model(model_path)
    transform = get_transforms()
    
    # Open video source
    if isinstance(source, int) or source.isdigit():
        source = int(source)
        print(f"\n📹 Opening webcam {source}...")
    else:
        print(f"\n📹 Opening video: {source}")
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    if total_frames > 0:
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Setup video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"\n📁 Saving output to: {output_path}")
    
    # Initialize temporal smoother
    smoother = TemporalSmoother(
        window_size=TEMPORAL_WINDOW,
        min_consecutive=MIN_CONSECUTIVE,
        threshold=CONFIDENCE_THRESHOLD
    )
    
    # Statistics
    stats = {
        'total_frames': 0,
        'accidents': 0,
        'raw_accidents': 0,
    }
    
    use_tta = TTA_ENABLED
    frame_num = 0
    prev_time = time.time()
    current_fps = fps
    
    print("\n🎬 Starting detection...")
    print("   Press 'Q' to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            stats['total_frames'] = frame_num
            
            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - prev_time
            if elapsed > 0:
                current_fps = 1.0 / elapsed
            prev_time = current_time
            
            # Predict
            confidence, individual_preds = predict(
                model, frame, device, transform, use_tta=use_tta
            )
            
            # Temporal smoothing
            is_confirmed, is_raw, recent_count = smoother.update(confidence)
            avg_confidence = smoother.get_avg_confidence()
            
            # Update stats
            if is_confirmed:
                stats['accidents'] += 1
            if is_raw:
                stats['raw_accidents'] += 1
            
            # Draw overlay
            display_frame = draw_overlay(
                frame.copy(), is_confirmed, is_raw, confidence, avg_confidence,
                recent_count, frame_num, total_frames, current_fps, stats
            )
            
            # Write output
            if writer:
                writer.write(display_frame)
            
            # Show display
            if show_display:
                cv2.imshow('Accident Detection', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️ Stopped by user")
                    break
                elif key == ord('s'):
                    # Screenshot
                    screenshot_path = f"screenshot_{frame_num}.jpg"
                    cv2.imwrite(screenshot_path, display_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                elif key == ord('t'):
                    # Toggle TTA
                    use_tta = not use_tta
                    print(f"🔄 TTA {'enabled' if use_tta else 'disabled'}")
            
            # Progress for video files
            if total_frames > 0 and frame_num % 100 == 0:
                progress = frame_num / total_frames * 100
                print(f"   Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DETECTION SUMMARY")
    print("="*60)
    print(f"   Total Frames Processed: {stats['total_frames']}")
    print(f"   Confirmed Accidents: {stats['accidents']}")
    print(f"   Raw Accident Frames: {stats['raw_accidents']}")
    if stats['total_frames'] > 0:
        accident_rate = stats['accidents'] / stats['total_frames'] * 100
        print(f"   Accident Rate: {accident_rate:.2f}%")
    print("="*60)
    
    return stats


# ============================================================================
# SINGLE IMAGE DETECTION
# ============================================================================
def detect_image(image_path, model_path=None, output_path=None):
    """
    Run accident detection on a single image.
    
    Args:
        image_path: Path to input image
        model_path: Path to .pth model file
        output_path: Path to save annotated image (optional)
    """
    print("\n" + "="*60)
    print("🖼️ SINGLE IMAGE DETECTION")
    print("="*60)
    
    # Load model
    model, device = load_model(model_path)
    transform = get_transforms()
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    print(f"\n📷 Image: {image_path}")
    print(f"   Size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Predict
    confidence, individual_preds = predict(model, frame, device, transform, use_tta=True)
    
    is_accident = confidence >= CONFIDENCE_THRESHOLD
    
    # Results
    print(f"\n📊 Results:")
    print(f"   Accident Probability: {confidence*100:.2f}%")
    print(f"   Prediction: {'🚨 ACCIDENT' if is_accident else '✅ NORMAL'}")
    print(f"   TTA Predictions: {[f'{p*100:.1f}%' for p in individual_preds]}")
    
    # Annotate image
    h, w = frame.shape[:2]
    color = COLOR_DANGER if is_accident else COLOR_SAFE
    label = f"{'ACCIDENT' if is_accident else 'NORMAL'}: {confidence*100:.1f}%"
    
    cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2)
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"\n💾 Saved to: {output_path}")
    else:
        cv2.imshow('Detection Result', frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return {
        'is_accident': is_accident,
        'confidence': confidence,
        'predictions': individual_preds.tolist()
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Accident Detection System (PyTorch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam detection
  python detect_pytorch.py --source 0
  
  # Video file detection
  python detect_pytorch.py --source video.mp4
  
  # Single image
  python detect_pytorch.py --image accident.jpg
  
  # Save output video
  python detect_pytorch.py --source video.mp4 --output result.mp4
  
  # Custom model path
  python detect_pytorch.py --source 0 --model models/accident_detector_best.pth
        """
    )
    
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: webcam index (0,1), video file, or RTSP URL')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image to detect (overrides --source)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video/image path')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable live display (for headless mode)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Confidence threshold (default: 0.6)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable Test-Time Augmentation')
    
    args = parser.parse_args()
    
    # Update global settings
    global CONFIDENCE_THRESHOLD, TTA_ENABLED
    CONFIDENCE_THRESHOLD = args.threshold
    TTA_ENABLED = not args.no_tta
    
    try:
        if args.image:
            # Single image mode
            detect_image(args.image, args.model, args.output)
        else:
            # Video mode
            detect_video(
                args.source,
                model_path=args.model,
                output_path=args.output,
                show_display=not args.no_display
            )
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
