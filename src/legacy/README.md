# Legacy TensorFlow Files

This folder contains the original TensorFlow/Keras implementation files that have been superseded by PyTorch versions.

## Why Legacy?

TensorFlow on Windows lacks native CUDA/GPU support for recent versions, making training extremely slow (CPU-only). The project was migrated to PyTorch which has full CUDA 12.x support on Windows.

## Files

| File | Description | Replaced By |
|------|-------------|-------------|
| `train.py` | TensorFlow training script with 3-phase transfer learning | `train_pytorch.py` |
| `detect.py` | Basic TensorFlow video/image detection | `detect_pytorch.py` |
| `detect_improved.py` | TensorFlow detection with TTA + temporal smoothing | `detect_pytorch.py` |
| `detect_tta.py` | TensorFlow Test-Time Augmentation implementation | `detect_pytorch.py` |
| `test_dataset.py` | Interactive dataset testing (TensorFlow) | `test_dataset_pytorch.py` |
| `verify_model.py` | Model verification and overfitting checks (TensorFlow) | `verify_model_pytorch.py` |

## Note

These files are kept for reference and documentation purposes. They will work if you have a TensorFlow environment with a compatible model file (`.keras` or `.h5`), but the PyTorch versions in the parent directory are recommended for active development and deployment.

## Model Compatibility

- **Legacy TF files**: Require `.keras` or `.h5` model files
- **Current PyTorch files**: Require `.pth` model files

The trained PyTorch model (`models/accident_detector_best.pth`) is not compatible with these legacy scripts.
