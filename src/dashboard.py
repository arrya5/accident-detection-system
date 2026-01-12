"""
Accident Detection System - Professional Dashboard

A modern web-based dashboard for video upload, accident detection,
and email alert management.

Features:
- Video upload and processing
- Real-time accident detection with progress tracking
- Email alert configuration and sending
- Detection history and statistics
- Professional UI with dark theme

Author: Arya Bhardwaj
Date: January 2026
License: MIT
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import os
import sys
import tempfile
from datetime import datetime
import time
from collections import deque
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import base64
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #b8d4e8;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0 0 0.5rem 0;
        font-weight: 500;
    }
    
    .metric-card .value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card.success .value { color: #22c55e; }
    .metric-card.danger .value { color: #ef4444; }
    .metric-card.warning .value { color: #f59e0b; }
    .metric-card.info .value { color: #3b82f6; }
    
    /* Alert box */
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #dc2626;
        color: white;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #14532d 0%, #166534 100%);
        border: 1px solid #22c55e;
        color: white;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #713f12 0%, #854d0e 100%);
        border: 1px solid #f59e0b;
        color: white;
    }
    
    /* Detection item */
    .detection-item {
        background: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ef4444;
    }
    
    .detection-item h4 {
        color: #f8fafc;
        margin: 0 0 0.5rem 0;
    }
    
    .detection-item p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-safe { background: #22c55e; color: white; }
    .status-accident { background: #ef4444; color: white; }
    .status-processing { background: #3b82f6; color: white; }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #0f172a;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #334155;
        border-radius: 12px;
        background: #1e293b;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Video frame styling */
    .frame-container {
        border: 2px solid #334155;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.85
TEMPORAL_WINDOW = 7
MIN_CONSECUTIVE = 5


# ============================================================================
# MODEL DEFINITION
# ============================================================================
class AccidentDetector(nn.Module):
    """MobileNetV2-based accident detection model."""
    
    def __init__(self, num_classes=1, pretrained=False):
        super(AccidentDetector, self).__init__()
        self.backbone = models.mobilenet_v2(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
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
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained PyTorch model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_paths = [
        "models/accident_detector_best.pth",
        "../models/accident_detector_best.pth",
        "models/accident_detector.pth",
        "../models/accident_detector.pth",
    ]
    
    for path in model_paths:
        if os.path.exists(path):
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
            return model, device
    
    return None, device


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


def predict_frame(model, frame, transform, device):
    """Predict accident probability for a single frame."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    tensor = transform(frame_rgb).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(tensor)
        probability = torch.sigmoid(output).item()
    
    return probability


def send_email_alert(config, frame, incident_id, confidence, timestamp):
    """Send email alert with accident screenshot."""
    try:
        msg = MIMEMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = config['recipient_email']
        msg['Subject'] = f"🚨 ACCIDENT ALERT - Incident #{incident_id} - {config.get('camera_location', 'Unknown Location')}"
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; background: #f8fafc;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); color: white; padding: 20px; text-align: center;">
                    <h1 style="margin: 0; font-size: 24px;">🚨 ACCIDENT DETECTED</h1>
                </div>
                
                <div style="padding: 25px;">
                    <h2 style="color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">Incident Details</h2>
                    
                    <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                        <tr>
                            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; color: #64748b;"><strong>Incident ID:</strong></td>
                            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; color: #1e293b;">#{incident_id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; color: #64748b;"><strong>Location:</strong></td>
                            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; color: #1e293b;">{config.get('camera_location', 'Unknown')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; color: #64748b;"><strong>Timestamp:</strong></td>
                            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; color: #1e293b;">{timestamp}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; color: #64748b;"><strong>Confidence:</strong></td>
                            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; color: #1e293b;">{confidence*100:.1f}%</td>
                        </tr>
                    </table>
                    
                    <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 15px; margin: 20px 0;">
                        <strong style="color: #92400e;">⚠️ IMMEDIATE ACTION REQUIRED</strong>
                        <p style="color: #92400e; margin: 10px 0 0 0;">Please dispatch emergency services to the location immediately.</p>
                    </div>
                    
                    <h3 style="color: #1e293b;">📸 Accident Screenshot</h3>
                    <p style="color: #64748b;">See attached image for visual confirmation.</p>
                </div>
                
                <div style="background: #f1f5f9; padding: 15px; text-align: center; border-top: 1px solid #e2e8f0;">
                    <p style="color: #64748b; font-size: 12px; margin: 0;">
                        Automated alert from Accident Detection System | AI Model: MobileNetV2 (99.80% accuracy)
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach screenshot
        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_data = img_encoded.tobytes()
        image = MIMEImage(img_data, name=f'accident_incident_{incident_id}.jpg')
        msg.attach(image)
        
        # Send email
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(msg)
        
        return True, "Email sent successfully!"
    
    except Exception as e:
        return False, str(e)


def draw_detection_overlay(frame, is_accident, confidence, frame_num, incident_count):
    """Draw detection status overlay on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Status bar at top
    status_color = (0, 0, 220) if is_accident else (0, 180, 0)
    status_text = "🚨 ACCIDENT DETECTED" if is_accident else "✓ NORMAL"
    
    cv2.rectangle(overlay, (0, 0), (w, 60), status_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Info bar at bottom
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, f"Frame: {frame_num}", (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Incidents: {incident_count}", (w - 150, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return frame


# ============================================================================
# MAIN DASHBOARD
# ============================================================================
def main():
    # Initialize session state
    if 'detections' not in st.session_state:
        st.session_state.detections = []
    if 'total_processed' not in st.session_state:
        st.session_state.total_processed = 0
    if 'email_sent_count' not in st.session_state:
        st.session_state.email_sent_count = 0
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚨 Accident Detection System</h1>
        <p>AI-Powered Video Analysis | Real-time Detection | Automated Alerts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Model Status
        model, device = load_model()
        if model:
            st.success(f"✅ Model Loaded ({device})")
            if device.type == 'cuda':
                st.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.error("❌ Model not found!")
            st.stop()
        
        st.markdown("---")
        
        # Detection Settings
        st.markdown("### 🎯 Detection Settings")
        threshold = st.slider("Confidence Threshold", 0.5, 0.99, 0.85, 0.01)
        temporal_window = st.slider("Temporal Window (frames)", 3, 15, 7)
        min_consecutive = st.slider("Min Consecutive Frames", 2, 10, 5)
        
        st.markdown("---")
        
        # Email Configuration
        st.markdown("### 📧 Email Alert Settings")
        email_enabled = st.toggle("Enable Email Alerts", value=False)
        
        if email_enabled:
            sender_email = st.text_input("Sender Email", placeholder="your.email@gmail.com")
            sender_password = st.text_input("App Password", type="password", placeholder="Gmail App Password")
            recipient_email = st.text_input("Recipient Email", placeholder="safety@authority.com")
            camera_location = st.text_input("Camera Location", value="Traffic Camera 1 - Main Junction")
            
            email_config = {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': sender_email,
                'sender_password': sender_password,
                'recipient_email': recipient_email,
                'camera_location': camera_location
            }
        else:
            email_config = None
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Session Statistics")
        st.metric("Videos Processed", st.session_state.total_processed)
        st.metric("Total Incidents Detected", len(st.session_state.detections))
        st.metric("Emails Sent", st.session_state.email_sent_count)
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Video Upload & Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a video file for accident detection",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Supported formats: MP4, AVI, MOV, MKV, WEBM"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            # Video info
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            st.markdown(f"""
            <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong style="color: #f8fafc;">📊 Video Information</strong>
                <div style="display: flex; gap: 2rem; margin-top: 0.5rem; color: #94a3b8;">
                    <span>📐 {width}x{height}</span>
                    <span>🎞️ {total_frames} frames</span>
                    <span>⏱️ {duration:.1f}s</span>
                    <span>🎬 {fps} FPS</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Process button
            if st.button("🔍 Start Detection", type="primary", use_container_width=True):
                # Initialize detection
                cap = cv2.VideoCapture(video_path)
                transform = get_transforms()
                predictions = deque(maxlen=temporal_window)
                
                incident_count = 0
                incidents = []
                frame_num = 0
                consecutive_accident = 0
                last_incident_frame = -100
                
                # Progress and display
                progress_bar = st.progress(0, text="Initializing...")
                frame_placeholder = st.empty()
                status_placeholder = st.empty()
                
                detection_start = time.time()
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_num += 1
                    
                    # Predict
                    confidence = predict_frame(model, frame, transform, device)
                    predictions.append(confidence)
                    avg_confidence = np.mean(list(predictions))
                    
                    is_accident = avg_confidence > threshold
                    
                    # Count consecutive accident frames
                    if is_accident:
                        consecutive_accident += 1
                    else:
                        consecutive_accident = 0
                    
                    # Detect new incident
                    if consecutive_accident >= min_consecutive and (frame_num - last_incident_frame) > temporal_window * 2:
                        incident_count += 1
                        last_incident_frame = frame_num
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        incidents.append({
                            'id': incident_count,
                            'frame': frame_num,
                            'confidence': avg_confidence,
                            'timestamp': timestamp,
                            'screenshot': frame.copy()
                        })
                        
                        # Send email alert
                        if email_enabled and email_config and all(email_config.get(k) for k in ['sender_email', 'sender_password', 'recipient_email']):
                            success, msg = send_email_alert(
                                email_config, frame, incident_count, avg_confidence, timestamp
                            )
                            if success:
                                st.session_state.email_sent_count += 1
                    
                    # Draw overlay and display
                    display_frame = draw_detection_overlay(
                        frame.copy(), is_accident, avg_confidence, frame_num, incident_count
                    )
                    
                    # Convert to RGB for display
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Update display every 3 frames for performance
                    if frame_num % 3 == 0:
                        frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                        progress_bar.progress(frame_num / total_frames, text=f"Processing frame {frame_num}/{total_frames}")
                        
                        if is_accident:
                            status_placeholder.markdown(f"""
                            <div class="alert-box alert-danger">
                                <span style="font-size: 24px;">🚨</span>
                                <div>
                                    <strong>ACCIDENT DETECTED</strong>
                                    <p style="margin: 0; opacity: 0.8;">Confidence: {avg_confidence*100:.1f}%</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            status_placeholder.markdown(f"""
                            <div class="alert-box alert-success">
                                <span style="font-size: 24px;">✅</span>
                                <div>
                                    <strong>NORMAL TRAFFIC</strong>
                                    <p style="margin: 0; opacity: 0.8;">Confidence: {(1-avg_confidence)*100:.1f}%</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                cap.release()
                processing_time = time.time() - detection_start
                
                # Update session state
                st.session_state.total_processed += 1
                st.session_state.detections.extend(incidents)
                
                # Final status
                progress_bar.progress(1.0, text="Detection complete!")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); padding: 1.5rem; border-radius: 12px; margin: 1rem 0; color: white;">
                    <h3 style="margin: 0 0 1rem 0;">📊 Detection Summary</h3>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
                        <div>
                            <p style="margin: 0; opacity: 0.7; font-size: 0.9rem;">Total Frames</p>
                            <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{total_frames}</p>
                        </div>
                        <div>
                            <p style="margin: 0; opacity: 0.7; font-size: 0.9rem;">Processing Time</p>
                            <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{processing_time:.1f}s</p>
                        </div>
                        <div>
                            <p style="margin: 0; opacity: 0.7; font-size: 0.9rem;">Incidents Detected</p>
                            <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: {'#ef4444' if incident_count > 0 else '#22c55e'};">{incident_count}</p>
                        </div>
                        <div>
                            <p style="margin: 0; opacity: 0.7; font-size: 0.9rem;">Processing Speed</p>
                            <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{total_frames/processing_time:.1f} FPS</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show detected incidents
                if incidents:
                    st.markdown("### 🚨 Detected Incidents")
                    for incident in incidents:
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            screenshot_rgb = cv2.cvtColor(incident['screenshot'], cv2.COLOR_BGR2RGB)
                            st.image(screenshot_rgb, caption=f"Incident #{incident['id']}", use_container_width=True)
                        with col_b:
                            st.markdown(f"""
                            <div class="detection-item">
                                <h4>🚨 Incident #{incident['id']}</h4>
                                <p><strong>Frame:</strong> {incident['frame']}</p>
                                <p><strong>Confidence:</strong> {incident['confidence']*100:.1f}%</p>
                                <p><strong>Timestamp:</strong> {incident['timestamp']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Manual email send button
                            if email_enabled and email_config:
                                if st.button(f"📧 Send Alert for Incident #{incident['id']}", key=f"email_{incident['id']}"):
                                    success, msg = send_email_alert(
                                        email_config, incident['screenshot'], incident['id'],
                                        incident['confidence'], incident['timestamp']
                                    )
                                    if success:
                                        st.success(f"✅ {msg}")
                                        st.session_state.email_sent_count += 1
                                    else:
                                        st.error(f"❌ Failed: {msg}")
                
                # Cleanup
                os.unlink(video_path)
    
    with col2:
        st.markdown("### 📋 Detection History")
        
        if st.session_state.detections:
            for detection in reversed(st.session_state.detections[-10:]):
                st.markdown(f"""
                <div class="detection-item">
                    <h4>Incident #{detection['id']}</h4>
                    <p>Frame: {detection['frame']} | Confidence: {detection['confidence']*100:.1f}%</p>
                    <p style="font-size: 0.8rem;">{detection['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No incidents detected yet. Upload a video to begin.")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.detections = []
            st.session_state.total_processed = 0
            st.session_state.email_sent_count = 0
            st.rerun()
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.markdown(f"""
            <div class="metric-card danger">
                <h3>Total Incidents</h3>
                <p class="value">{len(st.session_state.detections)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown(f"""
            <div class="metric-card info">
                <h3>Emails Sent</h3>
                <p class="value">{st.session_state.email_sent_count}</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
