# Architecture Document - Real-Time Accident Detection System

### Team Members
- **Arrya Thakur**
- **Chand Mehar**

---

## 1. Application Architecture

### Selected: **Monolithic Architecture**

All components (video capture, inference, alerting) run in a **single Python process**. This is ideal for real-time processing with low latency, shared GPU context, and simple deployment.

---

### 1.1 Microservices

Microservices architecture splits an application into small, independent services that communicate over a network (e.g., REST APIs).

**Why not suitable?**
- Adds network latency between services - harmful for real-time frame processing
- Unnecessary complexity for a single-pipeline system
- Overhead of managing multiple services for a small team

---

### 1.2 Event-Driven

Event-Driven architecture uses events (messages/triggers) to communicate between loosely coupled components via message brokers (e.g., Kafka, RabbitMQ).

**Why not suitable?**
- Only the email alert is event-based; the rest is a sequential pipeline
- Message broker adds overhead and latency to frame-by-frame processing
- Full event-driven is overkill for this use case

---

### 1.3 Serverless

Serverless architecture runs code in stateless, short-lived functions triggered on demand (e.g., AWS Lambda, Google Cloud Functions).

**Why not suitable?**
- Requires persistent GPU access - serverless doesn't support long-running GPU tasks
- Video stream is stateful and continuous - doesn't fit cold-start model
- High cost for continuous real-time inference workloads

---

## 2. Diagrams

### 2.1 Use Case Diagram

```
+------------------------------------------+
|       Accident Detection System          |
|                                          |
|  ( Capture Video Feed )     <--- [CCTV Camera]
|  ( Detect Accident )        <--- [Operator]
|  ( Send Email Alert )       ---> [Safety Authority]
|  ( Train Model )            <--- [ML Engineer]
+------------------------------------------+
```

### 2.2 Class Diagram

```
+-------------------+       +---------------------+
| VideoProcessor    |------>| AccidentDetector     |
| + read_frame()    |       | + predict(frame)     |
+-------------------+       | + tta_predict(frame) |
                            +---------------------+
                                     |
                                     v
                            +---------------------+
                            | TemporalSmoother    |
                            | + smooth(prediction)|
                            +---------------------+
                                     |
                                     v
                            +---------------------+
                            | EmailAlertSystem    |
                            | + send_alert(image) |
                            +---------------------+
```

### 2.3 Data Flow Diagram (DFD)

```
[CCTV] -> Capture Frame -> Preprocess -> Model Inference -> Temporal Smoothing -> Display
                                                                  |
                                                          [if Accident]
                                                                  |
                                                                  v
                                                    Send Email Alert -> [Safety Authority]
```

### 2.4 Component Diagram

```
+=============================================+
|  Video Capture (OpenCV)                     |
|       |                                     |
|       v                                     |
|  Preprocessing (torchvision transforms)     |
|       |                                     |
|       v                                     |
|  Inference Engine (MobileNetV2 + TTA)       |
|       |                                     |
|       v                                     |
|  Temporal Smoother                          |
|       |                                     |
|       v                                     |
|  Display Output    |    Email Alert (SMTP)  |
+=============================================+
```

### 2.5 Sequence Diagram

```
Operator -> VideoProcessor -> AccidentDetector -> TemporalSmoother -> Display
                                                      |
                                              [if Accident]
                                                      |
                                                      v
                                              EmailAlertSystem -> Safety Authority
                                              (loop for each frame)
```

### 2.6 Deployment Diagram

```
+---------------------------+
|  GPU Workstation          |
|  - Python 3.x            |
|  - PyTorch + CUDA        |
|  - OpenCV                |
|  - Model (.pth file)     |
+---------------------------+
      |               |
      v               v
[CCTV Camera]   [Gmail SMTP] -> [Safety Authority]
```

---

## 3. Database

### 3.1 ER Diagram

```
+-------------+  1:N  +------------------+
|   Camera    |------>| Detection_Log    |
| camera_id   |       | log_id           |
| location    |       | camera_id (FK)   |
+-------------+       | timestamp        |
                      | prediction       |
                      | confidence       |
                      | frame_path       |
                      +------------------+
```

### 3.2 Schema Design

```sql
CREATE TABLE Detection_Log (
    log_id        INT PRIMARY KEY AUTO_INCREMENT,
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
    prediction    ENUM('Accident', 'Normal') NOT NULL,
    confidence    FLOAT NOT NULL,
    frame_path    VARCHAR(500)
);
```

---

## 4. Data Exchange Contract

### 4.1 Frequency of Data Exchanges

| Exchange | Frequency |
|----------|-----------|
| Video frames (CCTV -> System) | Continuous, 25-30 FPS |
| Model inference | Per frame |
| Email alert | On accident detection only |
| Model loading | Once at startup |

### 4.2 Data Sets

| Data Set | Format | Size |
|----------|--------|------|
| Raw Video Frame | NumPy array (HxWx3) | ~1-3 MB |
| Preprocessed Tensor | Tensor (3x224x224) | ~600 KB |
| Prediction Result | {label, confidence} | ~50 bytes |
| Alert Payload | Email + JPEG screenshot | ~200-500 KB |
| Training Dataset | JPEG images (Accident/Normal folders) | ~13,000 images |
| Trained Model | .pth file | ~10-15 MB |

### 4.3 Mode of Exchanges

| Exchange | Mode |
|----------|------|
| CCTV -> System | Direct I/O (OpenCV VideoCapture) |
| Frame processing pipeline | In-Memory (Python/PyTorch) |
| CPU -> GPU | CUDA Memory Transfer |
| Accident alert | SMTP API (Gmail) |
| Model file loading | File I/O (torch.load) |

---