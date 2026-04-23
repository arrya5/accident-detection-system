# Real-Time Accident Detection: Project Workflow & Script Guide

This document provides a comprehensive walkthrough of the entire project, explaining the purpose of each script and how they contribute to the final result.

## Table of Contents
1. [Project Goal](#1-project-goal)
2. [The 3 Core Phases](#2-the-3-core-phases)
3. [Directory Structure Overview](#3-directory-structure-overview)
4. [Phase 1: Data Preparation & Management](#4-phase-1-data-preparation--management)
    - [`data/`](#data)
    - [`src/extract_frames.py`](#srcextract_framespy)
    - [`src/prepare_dataset.py`](#srcprepare_datasetpy)
5. [Phase 2: Model Training & Evaluation](#5-phase-2-model-training--evaluation)
    - [`src/train_pytorch.py`](#srctrain_pytorchpy)
    - [`src/test_dataset_pytorch.py`](#srctest_dataset_pytorchpy)
    - [`src/verify_model_pytorch.py`](#srcverify_model_pytorchpy)
6. [Phase 3: Real-Time Detection (The Demo)](#6-phase-3-real-time-detection-the-demo)
    - [`src/detect_pytorch.py`](#srcdetect_pytorchpy)
7. [How We Ran the Final Demo](#7-how-we-ran-the-final-demo)

---

## 1. Project Goal

The primary objective is to create a system that can **automatically detect road accidents in real-time** from a video feed (like a CCTV camera or a pre-recorded video) using a deep learning model.

---

## 2. The 3 Core Phases

The entire project can be broken down into three distinct phases:

1.  **Data Preparation**: Gathering and organizing thousands of images of "Accident" and "Non-Accident" scenes to create a dataset.
2.  **Model Training**: Using the prepared dataset to teach a deep learning model how to distinguish between an accident and a normal traffic scene. The best-performing model is saved.
3.  **Inference (Detection)**: Using the saved model to analyze new, unseen videos and run the live demo.

---

## 3. Directory Structure Overview

Before diving into the scripts, it's important to understand how the project is organized:

-   **`data/`**: Holds all the images used for training and testing the model.
-   **`docs/`**: Contains all project documentation, including this workflow, the architecture document, and methodology.
-   **`models/`**: This is where the final, trained deep learning models (`.pth` files) are stored.
-   **`output/`**: The default location for any files generated during a demo, such as incident screenshots or processed videos.
-   **`src/`**: The "source" folder, containing all the Python scripts that make the project work.

---

## 4. Phase 1: Data Preparation & Management

This phase is all about the data. A deep learning model is only as good as the data it's trained on.

### `data/`

This is the most critical directory for training. It's structured in a specific way that PyTorch understands:

```
data/
├── train/
│   ├── Accident/
│   │   ├── img1.jpg
│   │   └── ...
│   └── Non Accident/
│       ├── img1000.jpg
│       └── ...
├── val/
│   ├── Accident/
│   └── Non Accident/
└── test/
    ├── Accident/
    └── Non Accident/
```

-   **`train/`**: Contains the majority of the images (70%) used to teach the model.
-   **`val/`** (Validation): Contains a smaller set of images (15%) used during training to check the model's performance on data it hasn't been trained on. This helps prevent "overfitting."
-   **`test/`**: Contains a final set of images (15%) that the model has never seen. This is used for the final, unbiased evaluation of the model's real-world performance.

### `src/extract_frames.py`

-   **Purpose**: A utility script to help create a dataset from videos.
-   **Function**: You can point it to a folder of videos, and it will automatically extract individual frames (pictures) from them and save them as JPG files. This is useful if your source data is in video format.

### `src/prepare_dataset.py`

-   **Purpose**: To automatically split a large folder of images into the `train`, `val`, and `test` sets.
-   **Function**: You give it a source folder containing all your 'Accident' and 'Non Accident' images, and it will randomly shuffle and copy them into the correct `data/` subdirectories according to the 70-15-15 split. This ensures the training, validation, and testing sets are properly separated.

---

## 5. Phase 2: Model Training & Evaluation

This is where the "learning" happens. We use the prepared data to train the model and then test how well it performs.

### `src/train_pytorch.py`

-   **Purpose**: **This is the main training script.** It's the most important script for creating the model.
-   **Function**:
    1.  It loads the images from the `data/` directory.
    2.  It defines the model architecture (MobileNetV2 with a custom head).
    3.  It runs the **3-Phase Progressive Fine-Tuning** process, which is a special technique to make the model learn effectively.
    4.  During training, it saves logs and metrics to the `logs/` directory.
    5.  At the end of the training, it saves the best-performing model weights to a file like **`models/accident_detector_best.pth`**.

### `src/test_dataset_pytorch.py`

-   **Purpose**: A simple script for doing a quick test of the model on a few images from the test set.
-   **Function**: It loads the trained model and runs it on a few sample images to give you a quick, visual confirmation that the model is working as expected.

### `src/verify_model_pytorch.py`

-   **Purpose**: To conduct a final, formal evaluation of the trained model.
-   **Function**:
    1.  Loads the best trained model from the `models/` folder.
    2.  Runs the model on **every single image** in the `data/test` folder.
    3.  It calculates the final, official performance metrics (Accuracy, Precision, Recall, F1-Score).
    4.  It saves these results in a structured JSON file, like **`output/verification_report.json`**. This report is the proof of the model's performance (e.g., 99.80% accuracy).

---

## 6. Phase 3: Real-Time Detection (The Demo)

This is the final and most visible part of the project, where we use the trained model to perform a real-world task.

### `src/detect_pytorch.py`

-   **Purpose**: **This is the main demo script.** It runs the live accident detection on a video source.
-   **Function**:
    1.  Loads the trained model from `models/accident_detector.pth`.
    2.  Takes a video source as input (this can be a webcam, a video file path, or a network stream).
    3.  Processes the video frame-by-frame in real-time.
    4.  For each frame, it uses the model to predict the probability of an accident.
    5.  It applies **Temporal Smoothing** (analyzing a window of 7 frames) to reduce false alarms.
    6.  It displays the video feed along with a detailed dashboard showing the current status, confidence level, and other statistics.
    7.  When an accident is confirmed, it saves a screenshot of the incident to the **`output/`** folder.

---

## 7. How We Ran the Final Demo

To generate the results you saw, we performed the following steps:

1.  **Activated the Environment**: We first needed to activate the project's Python virtual environment, which contains all the necessary libraries (PyTorch, OpenCV, etc.).

2.  **Ran the Detection Script**: We then executed the `detect_pytorch.py` script using a specific command in the terminal.

The exact command used was:

```powershell
& ".venv\Scripts\Activate.ps1"; python "src/detect_pytorch.py" --source "c:\Users\arrya\Downloads\test_video.mp4" --model "models/accident_detector.pth"
```

-   **`python src/detect_pytorch.py`**: This tells Python to run our main demo script.
-   **`--source "c:\Users\arrya\Downloads\test_video.mp4"`**: This argument told the script to use the video file located at that specific path as its input.
-   **`--model "models/accident_detector.pth"`**: This argument explicitly told the script which trained model file to load for the detection.

This command initiated the process, loaded the model, and began analyzing the `test_video.mp4` file. The script then detected 3 distinct incidents and, as a result, saved the corresponding screenshots (`incident_1_frame_146.jpg`, etc.) into the `output` folder, which is the final result of the demonstration.
