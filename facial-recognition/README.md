# Facial Recognition Attendance System with Emotion & Liveness Detection

A complete face recognition system for attendance tracking. The system uses two different approaches for face verification (classification and metric learning), includes anti-spoofing to detect fake faces, and emotion detection to identify emotional states.

## Project Overview

This project implements an end-to-end face recognition attendance system with:

- **Face Verification**: Two approaches (Classification + Metric Learning)
- **Anti-Spoofing**: Detects fake faces (printed photos, screen photos)
- **Emotion Detection**: Identifies 7 emotional states from facial expressions
- **User Interface**: Web-based GUI for registration, verification, and attendance tracking

## System Architecture

```
User Interface (FastAPI Web App)
    ↓
Face Detection (MTCNN)
    ↓
Face Alignment & Preprocessing
    ↓
    ├─→ Anti-Spoofing Check → [Real/Spoofed]
    ├─→ Emotion Detection → [Happy/Sad/etc.]
    └─→ Face Embedding Extraction → [2048/512 dim vector]
            ↓
    Compare with Database (Cosine/Euclidean Distance)
            ↓
    [Match Found] or [No Match] or [Register New]
```

### Key Components

1. **Face Processing Pipeline**: Face detection, alignment, and preprocessing
2. **Face Verification Module**:
   - Classification Model: ResNet50 + Classification Head
   - Metric Learning Model: ResNet50 + Triplet Loss
3. **Database Storage**: Pickle format for storing face embeddings

## Quick Start

### 1. Setup Environment

```bash
# Activate conda environment
conda activate cos30082

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure

```
facial-recognition/
├── classification_data/      # Dataset (train_data/, val_data/, test_data/)
├── verification_data/         # Verification images
├── verification_pairs_val.txt  # Verification pairs for evaluation
├── src/
│   ├── models/               # Face verification models
│   │   ├── classification_model.py
│   │   └── metric_learning_model.py
│   ├── modules/              # Anti-spoofing, emotion detection
│   │   ├── anti_spoofing.py
│   │   └── emotion_detection.py
│   ├── utils/                # Data loading, evaluation, utilities
│   │   ├── data_loader.py
│   │   ├── evaluation.py
│   │   ├── face_detector.py
│   │   ├── face_database.py
│   │   └── triplet_mining.py
│   ├── train_classification.py
│   ├── train_metric_learning.py
│   ├── evaluate_classification.py
│   ├── evaluate_metric_learning.py
│   └── app_fastapi.py         # FastAPI web application
├── models/                   # Saved trained models
├── results/                  # ROC curves, evaluation results
└── requirements.txt
```

### 3. Train Models

**Train Classification Model:**

```bash
python src/train_classification.py \
    --data_dir classification_data \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-4 \
    --save_dir models
```

**Train Metric Learning Model:**

```bash
python src/train_metric_learning.py \
    --data_dir classification_data \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-5 \
    --margin 1.0 \
    --embedding_dim 512 \
    --save_dir models
```

**Training Tips:**

- Use `screen` or `tmux` to keep training running if you disconnect
- Monitor GPU: `watch -n 1 nvidia-smi`
- On RTX 5090: Training takes 8-16 hours for classification, 12-24 hours for metric learning

### 4. Evaluate Models

**Evaluate Classification Model:**

```bash
python src/evaluate_classification.py \
    --model_path models/classification_best.pth \
    --pairs_file verification_pairs_val.txt \
    --data_dir verification_data \
    --num_classes 4000 \
    --save_dir results
```

**Evaluate Metric Learning Model:**

```bash
python src/evaluate_metric_learning.py \
    --model_path models/metric_learning_best.pth \
    --pairs_file verification_pairs_val.txt \
    --data_dir verification_data \
    --embedding_dim 512 \
    --save_dir results
```

### 5. Run User Interface

```bash
# Start FastAPI web application
python src/app_fastapi.py

# Open browser to: http://localhost:8501
```

The UI includes:

- **Registration Mode**: Register new people with face images
- **Verification Mode**: Verify if person is registered (with anti-spoofing and emotion detection)
- **Real-Time Tracking**: Continuous face detection and attendance logging

## Technology Stack

- **PyTorch**: Deep learning framework
- **ResNet50**: Pre-trained CNN backbone
- **MTCNN**: Face detection and alignment
- **FastAPI**: Web framework for user interface
- **FER**: Pre-trained emotion detection model
- **scikit-learn**: Evaluation metrics (ROC, AUC)

## Performance Results

### Classification Model

- **AUC (Cosine)**: 0.9249
- **AUC (Euclidean)**: 0.9249
- **Status**: Excellent performance, exceeds 0.85 target

### Metric Learning Model

- **AUC (Cosine)**: 0.8747
- **AUC (Euclidean)**: 0.8747
- **Status**: Good performance, meets 0.85 target

Both models were trained on RTX 5090 GPU (Vast.AI) with 4,000 classes and 380K+ images.

## Key Concepts

### Face Verification

- **Input**: Two face images
- **Output**: Similarity score (0-1)
- **Decision**: Same person if score > threshold (0.85 for production)

### Classification Approach

- Train CNN to classify faces by identity
- Extract embeddings from layer before softmax
- Compare embeddings using cosine/Euclidean distance

### Metric Learning Approach

- Train CNN with triplet loss
- Learn embedding space where similar faces are close
- Use hard negative mining for better triplets

### Evaluation Metrics

- **ROC Curve**: Shows performance at different thresholds
- **AUC Score**: Area under ROC curve (higher is better)
- **Threshold**: Chosen to balance true positives and false positives

## Training Configuration

### Classification Model

- Learning Rate: 1e-4
- Batch Size: 64
- Epochs: 20
- Optimizer: Adam
- Loss: Cross-entropy

### Metric Learning Model

- Learning Rate: 1e-5
- Batch Size: 32
- Epochs: 30
- Margin: 1.0
- Optimizer: Adam
- Loss: Triplet Loss

## Troubleshooting

### Training too slow

- Use cloud GPU (Vast.AI RTX 5090) for faster training
- Increase batch_size if you have more VRAM
- Use `screen` or `tmux` to keep training running if disconnected

### CUDA out of memory

- Reduce batch_size (e.g., 32 or 16)
- Use frozen backbone for faster training

### Model not learning

- Check learning rate (try 1e-3 or 1e-5)
- Verify data loading and labels are correct
- Train for more epochs

### Low AUC score

- Train for more epochs
- Try different learning rate
- Check if embeddings are normalized
- Verify verification pairs are correct

## Important Notes

1. **Face alignment is crucial** - Always use MTCNN for face detection and alignment
2. **Normalize embeddings** - L2 normalization improves performance
3. **Hard negative mining** - Essential for metric learning (random triplets don't work well)
4. **Threshold selection** - Production threshold (0.85) may differ from evaluation threshold
5. **Metric learning is sensitive** - Requires careful hyperparameter tuning to avoid model collapse

## References

- FaceNet: https://arxiv.org/abs/1503.03832
- ArcFace: https://arxiv.org/abs/1801.07698
- ResNet: https://arxiv.org/abs/1512.03385

## License

This project is for educational purposes only.
