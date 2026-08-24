# AI Briefing: Real-Time Road Accident Detection System

**Version:** 1.0
**Date:** 2025-12-27
**Author:** GitHub Copilot (on behalf of the project team)

---

## 1. Project Overview

### 1.1. Core Objective
The primary goal of this project is to develop and implement a **real-time, automated system for detecting road accidents** from video feeds (CCTV, dashcams). The system uses a deep learning model to classify video frames as either "Accident" or "Normal Traffic" and triggers an alert upon detection.


### 1.3. Keywords
- Accident Detection
- Deep Learning
- Computer Vision
- Transfer Learning
- MobileNetV2
- PyTorch
- Real-time Video Analysis
- Convolutional Neural Networks (CNN)
- Traffic Safety

---

## 2. System Architecture

### 2.1. Architectural Pattern: Monolithic
The system is designed as a **single, monolithic Python application**. All functionalities—video capture, preprocessing, model inference, temporal smoothing, and alerting—run within a single process.

**Justification:**
- **Low Latency:** Avoids network overhead inherent in microservices, which is critical for real-time frame-by-frame processing.
- **Simplicity:** A single codebase is easier to manage, deploy, and debug for a small team.
- **Shared Resources:** Efficiently utilizes a single GPU context for all processing steps.

### 2.2. Data Flow Diagram (DFD)
```
[Video Source (CCTV/File)] -> [Frame Capture (OpenCV)] -> [Preprocessing (PyTorch)] -> [Model Inference (MobileNetV2)] -> [Temporal Smoothing] -> [Decision Logic]
                                                                                                                                    |
                                                                                                                              (if Accident)
                                                                                                                                    |
                                                                                                                                    v
                                                                                                                      [Alerting System (Email)] -> [Safety Authority]
```

### 2.3. Core Components
- **`detect_pytorch.py`**: The main application entry point. It handles video I/O, runs the main detection loop, and orchestrates all other components.
- **`AccidentDetector` (Model Class)**: A PyTorch `nn.Module` that defines the neural network architecture.
- **`TemporalSmoother` (Class)**: A stateful class that maintains a sliding window of recent predictions to filter out noise and confirm incidents.
- **`EmailAlertSystem` (Class)**: Manages the sending of email alerts in a background thread to avoid blocking the detection loop.

---

## 3. Dataset

### 3.1. Source and Composition
- **Source:** A curated dataset from multiple sources, including Kaggle's "Accident Detection from CCTV Footage," YouTube, and dashcam archives.
- **Total Size:** 13,228 images.
- **Classes:** Perfectly balanced between "Accident" and "Non Accident".
- **Splits:**
    - **Training:** 9,258 images (70%)
    - **Validation:** 1,984 images (15%)
    - **Test:** 1,986 images (15%)

### 3.2. Data Augmentation
To enhance model generalization, the following augmentations are applied during training:
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter (Brightness, Contrast)
- Random Affine (Translate, Scale)
- Gaussian Blur

---

## 4. Model & Methodology

### 4.1. Model Architecture: MobileNetV2
- **Backbone:** **MobileNetV2**, pre-trained on the ImageNet dataset.
- **Reasoning:** MobileNetV2 was chosen for its excellent balance between high accuracy and computational efficiency (low parameter count, fast inference), making it ideal for real-time applications.
- **Custom Head:** The original classifier of MobileNetV2 is replaced with a custom classification head:
    - `Input (1280 features)`
    - `Dropout(0.5)`
    - `Dense(512) -> BatchNorm -> ReLU -> Dropout(0.4)`
    - `Dense(256) -> BatchNorm -> ReLU -> Dropout(0.3)`
    - `Dense(128) -> BatchNorm -> ReLU -> Dropout(0.2)`
    - `Dense(1) -> Sigmoid Output`

### 4.2. Training Strategy: 3-Phase Progressive Fine-Tuning
This strategy was employed to achieve stable convergence and high accuracy.
- **Phase 1: Feature Extraction**
    - **Action:** Freeze the MobileNetV2 backbone. Train only the custom classification head.
    - **Goal:** Allow the classifier to learn the new task without corrupting the powerful pre-trained features.
    - **Learning Rate:** High (e.g., `1e-3`).
- **Phase 2: Fine-Tuning Top Layers**
    - **Action:** Unfreeze the top 50 layers of the backbone and continue training them along with the classifier.
    - **Goal:** Adapt the more abstract, high-level features of the backbone to the specifics of accident detection.
    - **Learning Rate:** Medium (e.g., `1e-4`).
- **Phase 3: Final Polish**
    - **Action:** Unfreeze all layers of the model.
    - **Goal:** Make small, final adjustments to the entire network for optimal performance.
    - **Learning Rate:** Very low (e.g., `1e-5`).

### 4.3. Key Techniques for Robustness
- **Test-Time Augmentation (TTA):** During inference, the model predicts on multiple augmented versions of the same frame (original, flipped, brightened, etc.). The final prediction is the average of these individual predictions. This significantly improves stability and reduces the chance of a single odd frame causing a misclassification.
- **Temporal Smoothing:** A `deque` (double-ended queue) maintains a history of the last `N` frame predictions (e.g., `N=7`). An accident is only confirmed if at least `M` of the last `N` frames are classified as an accident (e.g., `M=5`). This effectively filters out fleeting false positives.

---

## 5. Implementation Details

### 5.1. Technology Stack
- **Language:** Python 3.12
- **Framework:** PyTorch 2.6.0
- **Libraries:** OpenCV (for video processing), NumPy, Scikit-learn (for metrics).
- **Environment:** CUDA 12.4 for GPU acceleration.

### 5.2. Key Scripts
- **`src/train_pytorch.py`**:
    - **Purpose:** Handles the entire model training process, including data loading, the 3-phase training loop, and saving the final model (`.pth` file) and performance metrics (`.json` files).
    - **Execution:** `python src/train_pytorch.py --data_path /path/to/dataset`
- **`src/detect_pytorch.py`**:
    - **Purpose:** The main script for real-time detection. It loads the trained model and processes a video source (webcam, file, or stream).
    - **Execution:** `python src/detect_pytorch.py --source /path/to/video.mp4 --model /path/to/model.pth`
- **`src/verify_model_pytorch.py`**:
    - **Purpose:** Evaluates the trained model against the test dataset to generate the final performance report, including the confusion matrix and classification metrics.

### 5.3. Project Structure
```
/
├── models/                 # Stores trained .pth and .keras model files
│   ├── accident_detector.pth
│   └── accident_detector_best.pth
├── data/                   # Contains train/val/test image datasets
├── src/                    # All Python source code
│   ├── train_pytorch.py
│   ├── detect_pytorch.py
│   └── ...
├── docs/                   # Project documentation
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   └── RESULTS.md
├── output/                 # Default directory for output videos and screenshots
├── logs/                   # Directory for training logs and metrics
└── README.md               # Project summary
```

---

## 6. Performance Metrics (Test Set)

- **Accuracy:** 95.5%
- **Precision:** 100.00% (Zero false positives)
- **Recall (Sensitivity):** 99.60% (Only 4 false negatives out of 993 accident images)
- **F1-Score:** 95.5%
- **Confusion Matrix:**
    - **True Positives (Accident):** 989
    - **True Negatives (Normal):** 993
    - **False Positives (Normal -> Accident):** 0
    - **False Negatives (Accident -> Normal):** 4

---

## 7. How to Run the Demo

1.  **Set up the environment:**
    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    .venv\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Run detection on a video file:**
    ```bash
    python src/detect_pytorch.py --source "C:\path\to\your\video.mp4" --model "models/accident_detector.pth"
    ```

3.  **Run with email alerts (requires Gmail App Password):**
    ```bash
    python src/detect_pytorch.py --source "video.mp4" --email --sender-email "your_email@gmail.com" --sender-password "your_app_password" --recipient-email "authority@example.com"
    ```
