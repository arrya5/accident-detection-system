# Experimental Results

## Training Environment

| Component | Specification |
|-----------|---------------|
| Framework | PyTorch 2.6.0 + CUDA 12.4 |
| GPU | NVIDIA RTX 4060 Laptop (8.6 GB VRAM) |
| Model | MobileNetV2 (ImageNet pretrained) |
| Dataset | 13,228 images (9,258 train / 1,984 val / 1,986 test) |
| Training Time | ~15 minutes (GPU accelerated) |

## 1. Training Progress

### 1.1 Phase 1: Classification Head Training (Backbone Frozen)

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|------------|----------|---------------|
| 1 | 83.96% | 93.40% | 0.3458 | 0.1805 | 1e-3 |
| 3 | 90.76% | 96.93% | 0.2242 | 0.1239 | 1e-3 |
| 7 | 93.24% | 97.13% | 0.1669 | 0.0792 | 1e-3 |
| 9 | 93.47% | **97.93%** | 0.1576 | 0.0618 | 1e-3 |
| 15 | 94.56% | 97.68% | 0.1367 | 0.0634 | 1e-3 |

**Best Epoch**: 9 (val_accuracy = 97.93%)

### 1.2 Phase 2: Fine-Tuning Top 50 Layers

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|------------|----------|---------------|
| 1 | 96.49% | 99.14% | 0.0936 | 0.0334 | 1e-4 |
| 2 | 97.83% | 99.45% | 0.0607 | 0.0179 | 1e-4 |
| 5 | 99.20% | 99.90% | 0.0250 | 0.0042 | 1e-4 |
| 10 | 99.49% | 99.95% | 0.0166 | 0.0024 | 1e-4 |
| 11 | 99.52% | **100.00%** | 0.0131 | 0.0010 | 1e-4 |
| 15 | 99.71% | 100.00% | 0.0088 | 0.0011 | 1e-4 |

**Best Epoch**: 11 (val_accuracy = 100.00%)

### 1.3 Phase 3: Final Polish (All Layers Trainable)

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|------------|----------|---------------|
| 1 | 99.60% | 99.95% | 0.0121 | 0.0014 | 1e-5 |
| 2 | 99.79% | 99.95% | 0.0089 | 0.0018 | 1e-5 |
| 5 | 99.73% | **100.00%** | 0.0088 | 0.0008 | 1e-5 |

**Best Epoch**: 5 (val_accuracy = 100.00%)

---

## 2. Final Model Performance

### 2.1 Test Set Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.80%** |
| Test Loss | 0.0057 |
| Total Test Images | 1,986 |
| Correct Predictions | 1,982 |

### 2.2 Confusion Matrix

```
                    Predicted
                    Accident     Non Accident
Actual Accident       989            4
Actual Non Accident     0          993
```

### 2.3 Detailed Metrics

| Metric | Value |
|--------|-------|
| **Precision (Accident)** | 100.00% |
| **Recall (Accident)** | 99.60% |
| **F1-Score** | 99.80% |
| **Specificity** | 100.00% |

### 2.4 Class-wise Performance

| Class | Samples | Correct | Accuracy |
|-------|---------|---------|----------|
| Accident | 993 | 989 | 99.60% |
| Non Accident | 993 | 993 | 100.00% |

---

## 3. Model Architecture

### 3.1 Network Structure

| Component | Details |
|-----------|---------|
| Backbone | MobileNetV2 (ImageNet pretrained) |
| Feature Extractor | 1280-dimensional features |
| Classifier Head | 1280 → 512 → 256 → 128 → 1 |
| Dropout Rates | 0.5, 0.4, 0.3, 0.2 (progressive) |
| Batch Normalization | After each hidden layer |
| Total Parameters | 3,045,889 |

### 3.2 Training Configuration

| Hyperparameter | Phase 1 | Phase 2 | Phase 3 |
|----------------|---------|---------|---------|
| Learning Rate | 1e-3 | 1e-4 | 1e-5 |
| Epochs | 15 | 15 | 5 |
| Backbone | Frozen | Top 50 unfrozen | All trainable |
| Optimizer | AdamW | AdamW | AdamW |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |
| LR Scheduler | Warmup + Cosine | Cosine | Cosine |
| Early Stopping | Patience=8 | Patience=8 | Patience=8 |

### 3.3 Data Augmentation

| Augmentation | Training | Validation/Test |
|--------------|----------|-----------------|
| Resize | 256×256 | 256×256 |
| Random Crop | 224×224 | Center 224×224 |
| Horizontal Flip | 50% | No |
| Rotation | ±15° | No |
| Color Jitter | Brightness, Contrast, Saturation | No |
| Normalize | ImageNet stats | ImageNet stats |

---

## 4. Dataset Statistics

### 4.1 Data Distribution

| Split | Accident | Non Accident | Total |
|-------|----------|--------------|-------|
| Training | 4,629 | 4,629 | 9,258 |
| Validation | 992 | 992 | 1,984 |
| Test | 993 | 993 | 1,986 |
| **Total** | **6,614** | **6,614** | **13,228** |

### 4.2 Dataset Source

- **Source**: Kaggle Accident Detection Dataset
- **Balance**: Perfectly balanced (50/50 class distribution)
- **Split Ratio**: 70% train, 15% validation, 15% test

---

## 5. Computational Performance

### 5.1 Training Time

| Phase | Epochs | Time |
|-------|--------|------|
| Phase 1 (Frozen) | 15 | ~5 min |
| Phase 2 (Top 50) | 15 | ~6 min |
| Phase 3 (Full) | 5 | ~3 min |
| **Total** | **35** | **~15 min** |

*Trained on: NVIDIA RTX 4060 Laptop GPU (8.6 GB VRAM)*

### 5.2 Inference Speed

| Hardware | FPS | Latency |
|----------|-----|---------|
| RTX 4060 GPU | ~200 | 5ms |
| CPU (i7) | ~50 | 20ms |

### 5.3 Resource Usage

| Resource | Training | Inference |
|----------|----------|-----------|
| GPU VRAM | ~3 GB | ~500 MB |
| RAM | ~4 GB | ~500 MB |
| Model Size | - | 12 MB |

---

## 6. Training Curves

### 6.1 Accuracy Progress

```
Accuracy Progress:
100% │                              ████████████
 99% │                         █████
 98% │                    █████
 97% │               █████
 95% │          █████
 90% │     █████
 85% │████
     └────────────────────────────────────────
       Phase 1 (15)  Phase 2 (15)  Phase 3 (5)
       
Legend: Training accuracy progression over epochs
```

### 6.2 Loss Reduction

```
Loss:
0.35 │████
0.20 │    ████
0.10 │        ████
0.05 │            ████
0.01 │                ████████████████████
     └────────────────────────────────────────
       Phase 1        Phase 2        Phase 3
```

---

## 7. Comparison with Literature

| Study | Method | Dataset Size | Accuracy | Notes |
|-------|--------|--------------|----------|-------|
| Ijjina et al. (2019) | VGG-16 | 1,500 | 78% | Heavy model |
| Singh & Mohan (2021) | Custom CNN | 3,000 | 82% | From scratch |
| Yao et al. (2022) | ResNet-50 | 5,000 | 89% | Larger dataset |
| **This Work** | **MobileNetV2** | **13,228** | **99.80%** | **Lightweight + GPU** |

### Key Improvements Over Previous Work

1. **Larger Dataset**: 13,228 images vs typical 1,500-5,000
2. **Better Accuracy**: 99.80% vs 78-89% in literature
3. **Lightweight Model**: MobileNetV2 (3M params) vs VGG-16 (138M params)
4. **Three-Phase Training**: Progressive unfreezing for stable convergence
5. **GPU Acceleration**: 15-minute training on RTX 4060

---

## 8. Summary

### 8.1 Key Achievements

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Test Accuracy | >90% | **99.80%** | ✅ Exceeded |
| Precision | >90% | **100.00%** | ✅ Exceeded |
| Recall | >90% | **99.60%** | ✅ Exceeded |
| F1-Score | >90% | **99.80%** | ✅ Exceeded |
| Real-time FPS | >30 | **200+** | ✅ Exceeded |
| Model Size | <50 MB | **12 MB** | ✅ Exceeded |

### 8.2 Error Analysis

| Error Type | Count | Percentage |
|------------|-------|------------|
| True Positives | 989 | 49.80% |
| True Negatives | 993 | 50.00% |
| False Positives | 0 | 0.00% |
| False Negatives | 4 | 0.20% |

**Only 4 misclassifications out of 1,986 test images!**

---

## 9. Confusion Matrix Visualization

```
                     Predicted
                 Accident    Non Accident
           ┌────────────┬────────────────┐
  Actual   │    989     │       4        │  Accident
  Accident │    (TP)    │     (FN)       │  (993 total)
           ├────────────┼────────────────┤
  Actual   │     0      │      993       │  Non Accident
  Non Acc  │    (FP)    │     (TN)       │  (993 total)
           └────────────┴────────────────┘

  Precision: 989/(989+0) = 100.00%
  Recall:    989/(989+4) = 99.60%
  F1-Score:  2 × (1.00 × 0.996)/(1.00 + 0.996) = 99.80%
```

---

## 10. Conclusion

The MobileNetV2-based accident detection system achieves:

1. **99.80% accuracy** on unseen test data (1,986 images)
2. **100% precision** - zero false positives
3. **99.60% recall** - only 4 missed detections
4. **Real-time performance** at 200+ FPS on GPU
5. **Compact model** at 12 MB (3M parameters)
6. **Balanced performance** across both classes

### Model Files

| File | Description |
|------|-------------|
| `models/accident_detector.pth` | Final trained model |
| `models/accident_detector_best.pth` | Best checkpoint (100% val acc) |
| `logs/20251226_192316/` | Training logs and metrics |

The model is suitable for deployment in traffic monitoring systems to assist in rapid accident detection and emergency response.
