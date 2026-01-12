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
- Incident counting (distinct accidents, not frames)
- Optional audio alerts
- Detection logging to file

Author: Arya Bhardwaj
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
import logging
from collections import deque
from datetime import datetime
import time

# Optional: audio alerts (Windows only)
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Email alerts
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import threading

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

# Audio alert settings
AUDIO_ENABLED = False         # Set via command line
ALERT_FREQUENCY = 1000        # Hz
ALERT_DURATION = 200          # ms
MIN_ALERT_INTERVAL = 3.0      # Minimum seconds between alerts

# Email alert settings
EMAIL_ENABLED = False         # Set via command line
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': '',       # Set via command line or config
    'sender_password': '',    # App password for Gmail
    'recipient_email': '',    # Safety authority email
    'camera_location': 'Camera 1 - Main Junction'  # Camera/location identifier
}
MIN_EMAIL_INTERVAL = 30.0     # Minimum seconds between email alerts (prevent spam)


# ============================================================================
# EMAIL ALERT SYSTEM
# ============================================================================
class EmailAlertSystem:
    """Handles sending email alerts with accident screenshots."""
    
    def __init__(self, config: dict):
        self.config = config
        self.last_email_time = 0
        self.email_count = 0
        self.enabled = False
        
        # Validate config
        if config.get('sender_email') and config.get('sender_password') and config.get('recipient_email'):
            self.enabled = True
            print("   📧 Email alerts: Configured")
        else:
            print("   📧 Email alerts: Not configured (missing credentials)")
    
    def can_send_alert(self) -> bool:
        """Check if enough time has passed since last email."""
        if not self.enabled:
            return False
        current_time = time.time()
        return (current_time - self.last_email_time) >= MIN_EMAIL_INTERVAL
    
    def send_alert(self, frame: np.ndarray, incident_id: int, confidence: float,
                   frame_num: int, timestamp: str = None):
        """
        Send email alert with accident screenshot (runs in background thread).
        
        Args:
            frame: The accident frame (OpenCV BGR image)
            incident_id: Unique incident number
            confidence: Detection confidence
            frame_num: Frame number where accident detected
            timestamp: Optional timestamp string
        """
        if not self.can_send_alert():
            return False
        
        self.last_email_time = time.time()
        self.email_count += 1
        
        # Run in background thread to not block detection
        thread = threading.Thread(
            target=self._send_email_thread,
            args=(frame.copy(), incident_id, confidence, frame_num, timestamp)
        )
        thread.daemon = True
        thread.start()
        
        return True
    
    def _send_email_thread(self, frame, incident_id, confidence, frame_num, timestamp):
        """Background thread to send email."""
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['sender_email']
            msg['To'] = self.config['recipient_email']
            msg['Subject'] = f"🚨 ACCIDENT ALERT - Incident #{incident_id} - {self.config['camera_location']}"
            
            # Email body with HTML formatting
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <div style="background: #ff4444; color: white; padding: 15px; border-radius: 8px;">
                    <h1 style="margin: 0;">🚨 ACCIDENT DETECTED</h1>
                </div>
                
                <div style="margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px;">
                    <h2>Incident Details</h2>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Incident ID:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">#{incident_id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Location:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{self.config['camera_location']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Timestamp:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{timestamp}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Detection Confidence:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{confidence*100:.1f}%</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Frame Number:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{frame_num}</td>
                        </tr>
                    </table>
                </div>
                
                <div style="margin-top: 20px;">
                    <h3>📸 Accident Screenshot</h3>
                    <p>See attached image for visual confirmation.</p>
                </div>
                
                <div style="margin-top: 30px; padding: 15px; background: #fff3cd; border-radius: 8px; border: 1px solid #ffc107;">
                    <strong>⚠️ IMMEDIATE ACTION REQUIRED</strong>
                    <p style="margin: 5px 0 0 0;">Please dispatch emergency services to the location immediately.</p>
                </div>
                
                <hr style="margin-top: 30px;">
                <p style="color: #666; font-size: 12px;">
                    This is an automated alert from the Accident Detection System.<br>
                    Detection powered by AI (MobileNetV2 - 99.80% accuracy)
                </p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach screenshot
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_data = img_encoded.tobytes()
            
            image = MIMEImage(img_data, name=f'accident_incident_{incident_id}.jpg')
            image.add_header('Content-ID', '<accident_image>')
            msg.attach(image)
            
            # Send email
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['sender_email'], self.config['sender_password'])
                server.send_message(msg)
            
            print(f"\n   📧 Email alert sent for Incident #{incident_id}")
            
        except Exception as e:
            print(f"\n   ❌ Email failed: {str(e)}")


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
# TEMPORAL SMOOTHING & INCIDENT TRACKING
# ============================================================================
class TemporalSmoother:
    """
    Reduces false positives using temporal smoothing.
    
    Only confirms accident if multiple consecutive frames show accident.
    Also tracks distinct incidents (not just frames).
    """
    
    def __init__(self, window_size=5, min_consecutive=3, threshold=0.6):
        self.window = deque(maxlen=window_size)
        self.min_consecutive = min_consecutive
        self.threshold = threshold
        self.confidence_history = deque(maxlen=window_size)
        
        # Incident tracking
        self.in_incident = False
        self.incident_count = 0
        self.incident_start_frame = None
        self.current_incident_frames = 0
        self.last_alert_time = 0
    
    def update(self, confidence, frame_num=0):
        """
        Update with new frame and return smoothed decision.
        
        Args:
            confidence: Accident probability (0-1)
            frame_num: Current frame number
            
        Returns:
            tuple: (confirmed_accident, raw_prediction, recent_count, new_incident)
        """
        raw_accident = confidence >= self.threshold
        self.window.append(raw_accident)
        self.confidence_history.append(confidence)
        
        recent_accidents = sum(self.window)
        confirmed_accident = recent_accidents >= self.min_consecutive
        
        # Track incidents (distinct accidents, not frames)
        new_incident = False
        if confirmed_accident:
            if not self.in_incident:
                # New incident started
                self.in_incident = True
                self.incident_count += 1
                self.incident_start_frame = frame_num
                self.current_incident_frames = 1
                new_incident = True
            else:
                self.current_incident_frames += 1
        else:
            if self.in_incident:
                # Incident ended
                self.in_incident = False
                self.current_incident_frames = 0
        
        return confirmed_accident, raw_accident, recent_accidents, new_incident
    
    def get_avg_confidence(self):
        """Get average confidence over the window."""
        if len(self.confidence_history) == 0:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def should_alert(self):
        """Check if enough time has passed for a new alert."""
        current_time = time.time()
        if current_time - self.last_alert_time >= MIN_ALERT_INTERVAL:
            self.last_alert_time = current_time
            return True
        return False


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
# VISUALIZATION - PROFESSIONAL DASHBOARD
# ============================================================================

# Dashboard configuration
DASHBOARD_WIDTH = 320  # Width of side panel
HEADER_HEIGHT = 70
SECTION_PADDING = 15
LINE_HEIGHT = 28

# Dashboard colors (BGR)
PANEL_BG = (40, 40, 40)         # Dark gray
PANEL_HEADER = (60, 60, 60)     # Slightly lighter gray
ACCENT_BLUE = (255, 180, 50)    # Accent color
ACCENT_GREEN = (100, 220, 100)  # Success green
ACCENT_RED = (80, 80, 255)      # Danger red
ACCENT_ORANGE = (80, 165, 255)  # Warning orange
TEXT_PRIMARY = (255, 255, 255)  # White
TEXT_SECONDARY = (180, 180, 180) # Gray
DIVIDER_COLOR = (80, 80, 80)    # Divider lines


def draw_progress_bar(img, x, y, width, height, value, max_value=1.0, 
                      bg_color=(60, 60, 60), fill_color=ACCENT_GREEN, 
                      border_color=(100, 100, 100)):
    """Draw a modern progress bar."""
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    
    # Fill
    fill_width = int(width * min(value / max_value, 1.0))
    if fill_width > 0:
        cv2.rectangle(img, (x, y), (x + fill_width, y + height), fill_color, -1)
    
    # Border with rounded corners effect
    cv2.rectangle(img, (x, y), (x + width, y + height), border_color, 1)
    
    return img


def draw_circular_gauge(img, center_x, center_y, radius, value, max_value=1.0,
                        bg_color=(60, 60, 60), fill_color=ACCENT_GREEN):
    """Draw a circular gauge/arc for confidence."""
    # Background circle
    cv2.circle(img, (center_x, center_y), radius, bg_color, 3)
    
    # Calculate arc angle (270 degrees sweep, starting from bottom-left)
    angle = int(270 * min(value / max_value, 1.0))
    if angle > 0:
        cv2.ellipse(img, (center_x, center_y), (radius, radius), 
                    135, 0, angle, fill_color, 4)
    
    return img


def draw_metric_card(panel, x, y, width, height, title, value, unit="",
                     icon="", value_color=TEXT_PRIMARY):
    """Draw a metric card with title and value."""
    # Card background
    cv2.rectangle(panel, (x, y), (x + width, y + height), PANEL_HEADER, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), DIVIDER_COLOR, 1)
    
    # Title
    cv2.putText(panel, title, (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1)
    
    # Value
    value_text = f"{value}{unit}"
    cv2.putText(panel, value_text, (x + 10, y + height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, value_color, 2)


def create_dashboard(frame, is_confirmed, is_raw, confidence, avg_confidence,
                     recent_count, frame_num, total_frames, fps, stats,
                     use_tta=True, audio_enabled=False):
    """
    Create a professional dashboard with side panel showing all metrics.
    
    Returns:
        Combined frame with video and dashboard panel
    """
    h, w = frame.shape[:2]
    
    # Create dashboard panel
    panel = np.zeros((h, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    panel[:] = PANEL_BG
    
    # ===== HEADER =====
    cv2.rectangle(panel, (0, 0), (DASHBOARD_WIDTH, HEADER_HEIGHT), PANEL_HEADER, -1)
    cv2.putText(panel, "ACCIDENT DETECTION", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_BLUE, 2)
    cv2.putText(panel, "Real-time Monitoring System", (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1)
    
    # Divider
    cv2.line(panel, (0, HEADER_HEIGHT), (DASHBOARD_WIDTH, HEADER_HEIGHT), DIVIDER_COLOR, 1)
    
    y_offset = HEADER_HEIGHT + SECTION_PADDING
    
    # ===== STATUS SECTION =====
    if is_confirmed:
        status_text = "ACCIDENT DETECTED"
        status_color = ACCENT_RED
        status_bg = (40, 40, 80)
    elif is_raw:
        status_text = "POSSIBLE ACCIDENT"
        status_color = ACCENT_ORANGE
        status_bg = (40, 60, 80)
    else:
        status_text = "NORMAL TRAFFIC"
        status_color = ACCENT_GREEN
        status_bg = (40, 60, 40)
    
    # Status banner
    cv2.rectangle(panel, (10, y_offset), (DASHBOARD_WIDTH - 10, y_offset + 45), status_bg, -1)
    cv2.rectangle(panel, (10, y_offset), (DASHBOARD_WIDTH - 10, y_offset + 45), status_color, 2)
    cv2.putText(panel, status_text, (20, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    y_offset += 60
    
    # ===== CONFIDENCE SECTION =====
    cv2.putText(panel, "CONFIDENCE METRICS", (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_SECONDARY, 1)
    y_offset += 10
    cv2.line(panel, (15, y_offset), (DASHBOARD_WIDTH - 15, y_offset), DIVIDER_COLOR, 1)
    y_offset += 15
    
    # Current confidence with gauge
    gauge_color = ACCENT_RED if confidence > 0.6 else ACCENT_ORANGE if confidence > 0.4 else ACCENT_GREEN
    cv2.putText(panel, "Current", (20, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1)
    draw_progress_bar(panel, 80, y_offset, 140, 18, confidence, fill_color=gauge_color)
    cv2.putText(panel, f"{confidence*100:.1f}%", (230, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gauge_color, 1)
    y_offset += 30
    
    # Average confidence
    avg_color = ACCENT_RED if avg_confidence > 0.6 else ACCENT_ORANGE if avg_confidence > 0.4 else ACCENT_GREEN
    cv2.putText(panel, "Average", (20, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1)
    draw_progress_bar(panel, 80, y_offset, 140, 18, avg_confidence, fill_color=avg_color)
    cv2.putText(panel, f"{avg_confidence*100:.1f}%", (230, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, avg_color, 1)
    y_offset += 30
    
    # Threshold indicator
    cv2.putText(panel, "Threshold", (20, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1)
    cv2.putText(panel, f"{CONFIDENCE_THRESHOLD*100:.0f}%", (230, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_PRIMARY, 1)
    y_offset += 40
    
    # ===== DETECTION STATISTICS =====
    cv2.putText(panel, "DETECTION STATISTICS", (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_SECONDARY, 1)
    y_offset += 10
    cv2.line(panel, (15, y_offset), (DASHBOARD_WIDTH - 15, y_offset), DIVIDER_COLOR, 1)
    y_offset += 15
    
    # Metric cards in 2x2 grid
    card_w = 140
    card_h = 55
    
    # Incidents (distinct accidents)
    incident_color = ACCENT_RED if stats['incidents'] > 0 else ACCENT_GREEN
    draw_metric_card(panel, 10, y_offset, card_w, card_h, "INCIDENTS", 
                     stats['incidents'], value_color=incident_color)
    
    # Accident frames
    draw_metric_card(panel, 160, y_offset, card_w, card_h, "ACCIDENT FRAMES", 
                     stats['accident_frames'], value_color=ACCENT_ORANGE if stats['accident_frames'] > 0 else TEXT_PRIMARY)
    y_offset += card_h + 10
    
    # Total frames
    draw_metric_card(panel, 10, y_offset, card_w, card_h, "TOTAL FRAMES", 
                     stats['total_frames'], value_color=TEXT_PRIMARY)
    
    # Accident rate
    accident_rate = (stats['accident_frames'] / max(stats['total_frames'], 1)) * 100
    rate_color = ACCENT_RED if accident_rate > 10 else ACCENT_ORANGE if accident_rate > 5 else ACCENT_GREEN
    draw_metric_card(panel, 160, y_offset, card_w, card_h, "ACCIDENT RATE", 
                     f"{accident_rate:.1f}", unit="%", value_color=rate_color)
    y_offset += card_h + 20
    
    # ===== TEMPORAL ANALYSIS =====
    cv2.putText(panel, "TEMPORAL ANALYSIS", (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_SECONDARY, 1)
    y_offset += 10
    cv2.line(panel, (15, y_offset), (DASHBOARD_WIDTH - 15, y_offset), DIVIDER_COLOR, 1)
    y_offset += 15
    
    # Temporal window visualization
    cv2.putText(panel, f"Window: {recent_count}/{TEMPORAL_WINDOW} frames", (20, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_PRIMARY, 1)
    y_offset += 25
    
    # Visual representation of temporal window
    block_width = 50
    block_height = 20
    for i in range(TEMPORAL_WINDOW):
        x = 20 + i * (block_width + 5)
        if i < recent_count:
            color = ACCENT_RED
        else:
            color = (80, 80, 80)
        cv2.rectangle(panel, (x, y_offset), (x + block_width, y_offset + block_height), color, -1)
        cv2.rectangle(panel, (x, y_offset), (x + block_width, y_offset + block_height), (100, 100, 100), 1)
    y_offset += 40
    
    # ===== SYSTEM INFO =====
    cv2.putText(panel, "SYSTEM INFO", (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_SECONDARY, 1)
    y_offset += 10
    cv2.line(panel, (15, y_offset), (DASHBOARD_WIDTH - 15, y_offset), DIVIDER_COLOR, 1)
    y_offset += 15
    
    # FPS
    fps_color = ACCENT_GREEN if fps >= 25 else ACCENT_ORANGE if fps >= 15 else ACCENT_RED
    cv2.putText(panel, f"FPS: {fps:.1f}", (20, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
    
    # Progress (for video files)
    if total_frames > 0:
        progress = frame_num / total_frames
        cv2.putText(panel, f"Progress: {progress*100:.1f}%", (150, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_PRIMARY, 1)
    y_offset += 25
    
    # Frame counter
    frame_text = f"Frame: {frame_num}"
    if total_frames > 0:
        frame_text += f" / {total_frames}"
    cv2.putText(panel, frame_text, (20, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_SECONDARY, 1)
    y_offset += 35
    
    # ===== SETTINGS =====
    cv2.putText(panel, "SETTINGS", (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_SECONDARY, 1)
    y_offset += 10
    cv2.line(panel, (15, y_offset), (DASHBOARD_WIDTH - 15, y_offset), DIVIDER_COLOR, 1)
    y_offset += 15
    
    # TTA status
    tta_color = ACCENT_GREEN if use_tta else TEXT_SECONDARY
    tta_text = "ON" if use_tta else "OFF"
    cv2.putText(panel, f"TTA: {tta_text}", (20, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, tta_color, 1)
    
    # Audio status
    audio_color = ACCENT_GREEN if audio_enabled else TEXT_SECONDARY
    audio_text = "ON" if audio_enabled else "OFF"
    cv2.putText(panel, f"Audio: {audio_text}", (120, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, audio_color, 1)
    y_offset += 35
    
    # ===== CONTROLS (at bottom) =====
    controls_y = h - 60
    cv2.line(panel, (0, controls_y - 10), (DASHBOARD_WIDTH, controls_y - 10), DIVIDER_COLOR, 1)
    cv2.putText(panel, "CONTROLS", (15, controls_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_SECONDARY, 1)
    cv2.putText(panel, "[Q] Quit  [S] Screenshot", (15, controls_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_SECONDARY, 1)
    cv2.putText(panel, "[T] Toggle TTA  [A] Audio", (15, controls_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_SECONDARY, 1)
    
    # ===== ADD STATUS INDICATOR ON VIDEO FRAME =====
    video_frame = frame.copy()
    
    # Status banner on video
    banner_h = 50
    overlay = video_frame.copy()
    if is_confirmed:
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
    elif is_raw:
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 120, 180), -1)
    else:
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 120, 0), -1)
    cv2.addWeighted(overlay, 0.6, video_frame, 0.4, 0, video_frame)
    
    cv2.putText(video_frame, status_text, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Confidence on video
    cv2.putText(video_frame, f"{confidence*100:.0f}%", (w - 80, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Combine video and panel
    combined = np.hstack([video_frame, panel])
    
    return combined


def draw_overlay(frame, is_confirmed, is_raw, confidence, avg_confidence,
                 recent_count, frame_num, total_frames, fps, stats,
                 use_tta=True, audio_enabled=False):
    """
    Draw detection overlay on frame with professional dashboard.
    
    This is the main visualization function that creates the combined view
    with video feed and side panel dashboard.
    """
    return create_dashboard(
        frame, is_confirmed, is_raw, confidence, avg_confidence,
        recent_count, frame_num, total_frames, fps, stats,
        use_tta, audio_enabled
    )


# ============================================================================
# MAIN DETECTION LOOP
# ============================================================================
def detect_video(source, model_path=None, output_path=None, show_display=True,
                 enable_logging=False, log_file=None, email_config=None):
    """
    Run accident detection on a video source.
    
    Args:
        source: Video file path, webcam index (0, 1), or RTSP URL
        model_path: Path to .pth model file
        output_path: Path to save output video (optional)
        show_display: Whether to show live display
        enable_logging: Whether to log detections to file
        log_file: Path to log file (optional)
        email_config: Email configuration dict for alerts (optional)
    """
    global AUDIO_ENABLED, EMAIL_ENABLED
    
    print("\n" + "="*60)
    print("🚗 ACCIDENT DETECTION SYSTEM (PyTorch)")
    print("="*60)
    
    # Setup email alerts
    email_system = None
    if email_config and EMAIL_ENABLED:
        email_system = EmailAlertSystem(email_config)
    
    # Setup logging
    if enable_logging:
        log_path = log_file or f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("Detection session started")
    else:
        logger = None
    
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
    
    if AUDIO_ENABLED and AUDIO_AVAILABLE:
        print("   🔊 Audio alerts: Enabled")
    
    # Setup video writer if output path specified
    # Note: Output includes dashboard panel, so width is video_width + DASHBOARD_WIDTH
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_width = width + DASHBOARD_WIDTH  # Include dashboard panel
        writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
        print(f"\n📁 Saving output to: {output_path}")
        print(f"   Output resolution: {output_width}x{height} (includes dashboard)")
    
    # Initialize temporal smoother
    smoother = TemporalSmoother(
        window_size=TEMPORAL_WINDOW,
        min_consecutive=MIN_CONSECUTIVE,
        threshold=CONFIDENCE_THRESHOLD
    )
    
    # Statistics - now tracks incidents (distinct accidents) vs frames
    stats = {
        'total_frames': 0,
        'incidents': 0,         # Distinct accident incidents
        'accident_frames': 0,   # Total frames with confirmed accident
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
            
            # Temporal smoothing with incident tracking
            is_confirmed, is_raw, recent_count, new_incident = smoother.update(
                confidence, frame_num
            )
            avg_confidence = smoother.get_avg_confidence()
            
            # Update stats
            stats['incidents'] = smoother.incident_count
            if is_confirmed:
                stats['accident_frames'] += 1
            if is_raw:
                stats['raw_accidents'] += 1
            
            # Handle new incident
            if new_incident:
                if logger:
                    logger.warning(f"INCIDENT #{stats['incidents']} detected at frame {frame_num} "
                                 f"(confidence: {confidence*100:.1f}%)")
                
                # Audio alert
                if AUDIO_ENABLED and AUDIO_AVAILABLE and smoother.should_alert():
                    try:
                        winsound.Beep(ALERT_FREQUENCY, ALERT_DURATION)
                    except:
                        pass
                
                # Always save screenshot of incident for evidence/review
                screenshot_path = f"output/incident_{stats['incidents']}_frame_{frame_num}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"\n   📸 Incident #{stats['incidents']} screenshot saved: {screenshot_path}")
                
                # Email alert with screenshot (if configured)
                if email_system and email_system.enabled:
                    # Send email alert
                    email_system.send_alert(
                        frame=frame,
                        incident_id=stats['incidents'],
                        confidence=confidence,
                        frame_num=frame_num
                    )
            
            # Draw overlay with professional dashboard
            display_frame = draw_overlay(
                frame.copy(), is_confirmed, is_raw, confidence, avg_confidence,
                recent_count, frame_num, total_frames, current_fps, stats,
                use_tta=use_tta, audio_enabled=AUDIO_ENABLED
            )
            
            # Write output - need to resize back if saving without dashboard
            if writer:
                # Save with dashboard
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
                elif key == ord('a'):
                    # Toggle audio
                    AUDIO_ENABLED = not AUDIO_ENABLED
                    status = 'enabled' if AUDIO_ENABLED else 'disabled'
                    print(f"🔊 Audio alerts {status}")
            
            # Progress for video files
            if total_frames > 0 and frame_num % 100 == 0:
                progress = frame_num / total_frames * 100
                print(f"   Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if logger:
            logger.info(f"Detection session ended - {stats['incidents']} incidents detected")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DETECTION SUMMARY")
    print("="*60)
    print(f"   Total Frames Processed: {stats['total_frames']}")
    print(f"   Distinct Incidents: {stats['incidents']}")
    print(f"   Accident Frames: {stats['accident_frames']}")
    print(f"   Raw Accident Frames: {stats['raw_accidents']}")
    if stats['total_frames'] > 0:
        accident_rate = stats['accident_frames'] / stats['total_frames'] * 100
        print(f"   Accident Frame Rate: {accident_rate:.2f}%")
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
    
    # Create display with dashboard
    stats = {
        'total_frames': 1,
        'incidents': 1 if is_accident else 0,
        'accident_frames': 1 if is_accident else 0,
        'raw_accidents': 1 if is_accident else 0,
    }
    
    display_frame = draw_overlay(
        frame.copy(), is_accident, is_accident, confidence, confidence,
        1 if is_accident else 0, 1, 1, 0, stats,
        use_tta=True, audio_enabled=False
    )
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, display_frame)
        print(f"\n💾 Saved to: {output_path}")
    else:
        cv2.imshow('Detection Result', display_frame)
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
  
  # With audio alerts and logging
  python detect_pytorch.py --source 0 --audio --log
  
  # Custom model path
  python detect_pytorch.py --source 0 --model models/accident_detector_best.pth

Controls (during video playback):
  Q - Quit
  S - Screenshot
  T - Toggle TTA
  A - Toggle audio alerts
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
    parser.add_argument('--audio', action='store_true',
                        help='Enable audio alerts on accident detection')
    parser.add_argument('--log', action='store_true',
                        help='Enable logging detections to file')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file (default: auto-generated)')
    
    # Email alert arguments
    parser.add_argument('--email', action='store_true',
                        help='Enable email alerts on accident detection')
    parser.add_argument('--sender-email', type=str, default=None,
                        help='Sender Gmail address')
    parser.add_argument('--sender-password', type=str, default=None,
                        help='Sender Gmail app password (not regular password)')
    parser.add_argument('--recipient-email', type=str, default=None,
                        help='Recipient email (safety authority)')
    parser.add_argument('--camera-location', type=str, default='Camera 1 - Main Junction',
                        help='Camera/location identifier for alerts')
    
    args = parser.parse_args()
    
    # Update global settings
    global CONFIDENCE_THRESHOLD, TTA_ENABLED, AUDIO_ENABLED, EMAIL_ENABLED
    CONFIDENCE_THRESHOLD = args.threshold
    TTA_ENABLED = not args.no_tta
    AUDIO_ENABLED = args.audio
    EMAIL_ENABLED = args.email
    
    # Setup email config
    email_config = None
    if args.email:
        email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': args.sender_email,
            'sender_password': args.sender_password,
            'recipient_email': args.recipient_email,
            'camera_location': args.camera_location
        }
    
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
                show_display=not args.no_display,
                enable_logging=args.log,
                log_file=args.log_file,
                email_config=email_config
            )
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()