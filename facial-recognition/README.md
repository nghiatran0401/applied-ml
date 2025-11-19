# Facial Recognition Attendance System with Emotion & Liveness Detection

## 📋 Project Overview

Build an end-to-end face recognition attendance system for enterprise use with:

- **Face Verification**: Two approaches (Classification + Metric Learning)
- **Anti-Spoofing**: Detect fake faces (printed photos, screen photos)
- **Emotion Detection**: Identify emotional states from facial expressions
- **User Interface**: GUI for registration, verification, and system interaction

### Project Goal: Achieve 80% (80/100 marks)

---

## 🎯 Project Requirements

### 1. Face Verification (30 marks - Target: 24/30)

**Two approaches must be implemented:**

#### Approach 1: Classification-Based (Supervised Learning)

- Train CNN to classify faces by identity
- Use softmax layer during training
- Extract embeddings from layer before softmax
- Compare embeddings using distance metrics

#### Approach 2: Metric Learning (Self-Supervised)

- Train CNN with triplet loss
- Learn embedding space where similar faces are close
- Use hard negative mining for better triplets
- Compare embeddings using distance metrics

**Evaluation Metrics:**

- ROC curve and AUC score
- Test on `verification_pairs_val.txt`
- Compare cosine similarity vs Euclidean distance

### 2. Anti-Spoofing Module (15 marks - Target: 12/15)

- Detect spoofed/non-real faces (printed photos, screen photos)
- Integrate seamlessly with face recognition pipeline
- Can use pre-trained models or existing tools

### 3. Emotion Detection Module (10 marks - Target: 8/10)

- Identify emotional states (happy, sad, angry, surprise, fear, disgust, neutral)
- Display emotion with icons/labels in UI
- Can use pre-trained models (FER2013, FER+)

### 4. User Interface (10 marks - Target: 8/10)

- **Registration Mode**: Register new face IDs
- **Verification Mode**: Check if person is registered
- Display: Identity, Emotion, Liveness status, Confidence scores
- Use Streamlit/Gradio for easy GUI development

### 5. Project Report (20 marks - Target: 16/20)

- **Methodology Section**: System architecture, model details, training schemes, hyperparameters
- **Results & Discussion**: Performance comparison, best approach justification, limitations

### 6. Demo & Innovation (15 marks - Target: 12/15)

- Working demonstration
- Simple innovations (confidence scores, batch processing, etc.)

---

## 🏗️ System Architecture

```
User Interface (Streamlit)
    ↓
Face Detection (MTCNN/RetinaFace)
    ↓
Face Alignment & Preprocessing
    ↓
    ├─→ Anti-Spoofing Check → [Reject if spoofed]
    ├─→ Emotion Detection → [Happy/Sad/etc.]
    └─→ Face Embedding Extraction → [128/512 dim vector]
            ↓
    Compare with Database (Cosine/Euclidean Distance)
            ↓
    [Match] or [No Match] or [Register New]
```

### Key Components

1. **Face Processing Pipeline**

   - Face detection and alignment
   - Parallel processing: Anti-spoofing, Emotion, Embedding extraction

2. **Face Verification Module**

   - **Classification Model**: ResNet50/EfficientNet + Classification Head
   - **Metric Learning Model**: ResNet50/EfficientNet + Triplet Loss
   - **Similarity Computation**: Cosine similarity & Euclidean distance

3. **Database & Storage**
   - SQLite/JSON for storing face embeddings
   - Registered identities (ID, Name, Embedding)

---

## 🚀 Quick Start

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
├── verification_pairs_val.txt # Verification pairs for evaluation
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
│   │   ├── triplet_mining.py
│   │   └── create_subset.py
│   ├── train_classification.py
│   ├── train_metric_learning.py
│   ├── evaluate_classification.py
│   ├── evaluate_metric_learning.py
│   └── app.py                # Streamlit UI
├── models/                   # Saved trained models
├── results/                  # ROC curves, evaluation results
├── training.log              # Training logs
└── requirements.txt
```

### 3. Train Classification Model

**On Vast.AI (RTX 5090):**

```bash
# Connect via SSH or use Jupyter Terminal
cd facial-recognition

# Start training (with screen for detaching)
screen -S training
python3 src/train_classification.py \
    --data_dir classification_data \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-4 \
    --save_dir models

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r training
```

**Training Tips**:

- Use `screen` or `tmux` to keep training running if you disconnect
- Monitor GPU: `watch -n 1 nvidia-smi` (in another terminal)
- Check progress: `tail -f training.log` (if logging to file)
- Default settings: 20 epochs, batch_size 64 (optimized for RTX 5090)

**⚠️ Training Time Considerations:**

With **4,000 classes** and **380K+ images**:

**Training Platform: Vast.AI with RTX 5090**

- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **Device**: CUDA (auto-detected)
- **Batch Size**: 64 (optimized for 32GB VRAM)
- **Epochs**: 20 (default, can be adjusted)

**Training Time Estimates (RTX 5090):**

| Configuration     | Time per Epoch  | 20 Epochs      | Cost (at $0.37/hr) |
| ----------------- | --------------- | -------------- | ------------------ |
| **Full training** | **~0.4-0.8 hr** | **8-16 hours** | **$3-6**           |
| Frozen backbone   | ~0.2-0.4 hr     | 4-8 hours      | $1.50-3            |

**Current Configuration:**

- **Auto-detects CUDA** (RTX 5090)
- **Full training** (all layers unfrozen for best accuracy)
- **Batch size 64** (utilizes 32GB VRAM efficiently)
- **20 epochs** (default, provides good convergence)
- **Total cost**: ~$3-6 for complete training

**Alternative: Use Subset for Faster Training**

If training is too slow, create a subset:

```bash
# Create subset with 500 classes (much faster)
python src/utils/create_subset.py \
    --source classification_data/train_data \
    --target classification_data/train_data_subset \
    --num_classes 500

# Then train on subset
python src/train_classification.py \
    --data_dir classification_data_subset \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4
```

**Training Time Estimates:**

| Configuration                 | Time per Epoch | 10 Epochs                  | Feasible? |
| ----------------------------- | -------------- | -------------------------- | --------- |
| Full dataset, all layers      | ~63 hours      | ~630 hours (26 days)       | ❌ No     |
| Full dataset, frozen backbone | ~6-10 hours    | ~60-100 hours (2.5-4 days) | ⚠️ Maybe  |
| 500 classes, frozen backbone  | ~1-2 hours     | ~10-20 hours               | ✅ Yes    |
| 1000 classes, frozen backbone | ~2-4 hours     | ~20-40 hours               | ✅ Yes    |

### 4. Train Metric Learning Model

**On Vast.AI (RTX 5090):**

```bash
# Start training (with screen for detaching)
screen -S metric_training
python3 src/train_metric_learning.py \
    --data_dir classification_data \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-4 \
    --margin 0.5 \
    --embedding_dim 512 \
    --save_dir models

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r metric_training
```

### 5. Evaluate Models

**Evaluate Classification Model**:

```bash
python src/evaluate_classification.py \
    --model_path models/classification_best.pth \
    --pairs_file verification_pairs_val.txt \
    --data_dir verification_data \
    --num_classes 4000 \
    --save_dir results
```

**Evaluate Metric Learning Model**:

```bash
python src/evaluate_metric_learning.py \
    --model_path models/metric_learning_best.pth \
    --pairs_file verification_pairs_val.txt \
    --data_dir verification_data \
    --embedding_dim 512 \
    --save_dir results
```

### 6. Run User Interface

```bash
# Start Streamlit app
streamlit run src/app.py
```

The UI includes:

- **Registration Mode**: Register new people with face images
- **Verification Mode**: Verify if person is registered (with anti-spoofing and emotion detection)
- **Database View**: View all registered people

---

## 🛠️ Technology Stack

### Core ML Framework

- **PyTorch** (recommended) - Better for research, easier debugging
- **GPU Support**: CUDA (NVIDIA RTX 5090 on Vast.AI)
- **Device Selection**: Auto-detects CUDA when available

### Face Detection

- **MTCNN** or **RetinaFace** - Accurate, easy to use

### CNN Backbones

- **ResNet50** or **EfficientNet-B0** - Pre-trained weights available

### Anti-Spoofing

- **Silent-Face-Anti-Spoofing** (GitHub) or
- **DeepFace** library's liveness detection

### Emotion Detection

- **FER2013 pre-trained model** or
- **FER+ dataset + simple CNN**

### User Interface

- **Streamlit** (easiest) - Quick to build, good for ML demos

### Database

- **SQLite** or **JSON file** - Simple, no server needed

### Evaluation

- **scikit-learn** - ROC curve, AUC
- **matplotlib/seaborn** - Visualizations

---

## 📊 Success Metrics

### Minimum Viable Product (MVP) for 80%

1. ✅ Both face verification approaches implemented
2. ✅ AUC score > 0.85 for at least one approach
3. ✅ Anti-spoofing module working (detects obvious spoofs)
4. ✅ Emotion detection working (5+ emotions)
5. ✅ Functional GUI with all features
6. ✅ Clear project report with methodology and results

### Expected Performance

- **Training Accuracy**: > 90%
- **Validation Accuracy**: > 85%
- **Verification AUC (Cosine)**: > 0.85 (Excellent), > 0.75 (Good)
- **Verification AUC (Euclidean)**: > 0.80 (Excellent), > 0.70 (Good)

---

## 📝 Implementation Plan

### Phase 1: Setup & Data Preparation (Days 1-2)

- [ ] Environment setup
- [ ] Data loading & preprocessing
- [ ] Face detection and alignment
- [ ] Create data loaders

### Phase 2: Classification-Based Face Verification (Days 3-5)

- [ ] Build classification model (ResNet50 + classification head)
- [ ] Train with cross-entropy loss
- [ ] Extract embeddings from penultimate layer
- [ ] Evaluate on verification pairs (ROC/AUC)

### Phase 3: Metric Learning Face Verification (Days 6-9)

- [ ] Build triplet network
- [ ] Implement triplet loss
- [ ] Train with hard negative mining
- [ ] Evaluate on verification pairs (ROC/AUC)

### Phase 4: Anti-Spoofing Module (Days 10-12)

- [ ] Research and select approach (pre-trained model recommended)
- [ ] Integrate into pipeline
- [ ] Test with sample images

### Phase 5: Emotion Detection Module (Days 13-14)

- [ ] Use pre-trained FER2013 model
- [ ] Integrate into pipeline
- [ ] Display in UI

### Phase 6: User Interface (Days 15-17)

- [ ] Build Streamlit GUI
- [ ] Implement registration mode
- [ ] Implement verification mode
- [ ] Display results (identity, emotion, liveness)

### Phase 7: Integration & Testing (Days 18-19)

- [ ] End-to-end integration
- [ ] Test all workflows
- [ ] Performance optimization

### Phase 8: Evaluation & Comparison (Day 20)

- [ ] Compare classification vs metric learning
- [ ] Compare cosine vs Euclidean distance
- [ ] Generate ROC curves and performance metrics

### Phase 9: Project Report (Days 21-23)

- [ ] Methodology section
- [ ] Results & Discussion section
- [ ] Visualizations and performance tables

### Phase 10: Demo Preparation (Day 24)

- [ ] Prepare demo scenarios
- [ ] Test demo flow
- [ ] Add simple innovations

---

## 🎓 Key Concepts

### Face Verification

- **Input**: Two face images
- **Output**: Similarity score (0-1)
- **Decision**: Same person if score > threshold

### Classification Approach

- Train CNN to classify faces by identity
- Use layer before softmax as embedding
- Compare embeddings using cosine/Euclidean distance

### Metric Learning Approach

- Train CNN with triplet loss
- Learn embedding space where similar faces are close
- Compare embeddings using cosine/Euclidean distance

### Evaluation

- **ROC Curve**: Plot TPR vs FPR at different thresholds
- **AUC Score**: Area under ROC curve (higher is better)
- **Threshold**: Chosen to balance TPR and FPR

---

## 📚 Essential Resources

### Papers

- **FaceNet** (Triplet Loss): https://arxiv.org/abs/1503.03832
- **ArcFace** (Classification): https://arxiv.org/abs/1801.07698

### Datasets

- **FER2013** (Emotion): https://www.kaggle.com/datasets/msambare/fer2013

### Libraries

- `facenet-pytorch` - Face detection, alignment
- `fer` - Emotion detection
- `deepface` - Pre-built face recognition (use for reference only)

---

## ⚠️ Common Pitfalls to Avoid

1. **Don't skip data preprocessing** - Face alignment is crucial
2. **Don't forget to normalize embeddings** - L2 normalization improves performance
3. **Don't ignore triplet mining** - Hard negative mining is important for metric learning
4. **Don't overcomplicate UI** - Simple and functional is better than complex and broken
5. **Don't forget to compare approaches** - Must show comparison in report
6. **Don't skip evaluation** - Must compute ROC/AUC on provided validation set

---

## ⏱️ Training Time & Efficiency

### Current Dataset

- **Training images**: 380,638
- **Number of classes**: 4,000
- **Device**: CUDA (NVIDIA RTX 5090 on Vast.AI)

### GPU Acceleration (RTX 5090)

**Training Platform: Vast.AI Cloud GPU**

- **GPU**: NVIDIA GeForce RTX 5090
- **VRAM**: 32GB (allows batch_size=64-128)
- **Architecture**: Latest RTX 50-series (Blackwell)
- **Cost**: $0.37/hour (~$3-6 for complete training)

**Performance Benefits:**

- **RTX 5090**: 7-12x faster than local Mac M2 (MPS)
- **CUDA**: Optimized for PyTorch training
- **32GB VRAM**: Allows larger batch sizes for faster training
- **Full training**: All layers unfrozen for best accuracy

### Training Time Estimates (RTX 5090)

**With RTX 5090 (Full Training):**

- **Time per epoch**: ~0.4-0.8 hours
- **20 epochs**: ~8-16 hours total ✅
- **Cost**: ~$3-6 (at $0.37/hr)

**Comparison:**

| Platform     | Time per Epoch | 20 Epochs      | Cost     |
| ------------ | -------------- | -------------- | -------- |
| **RTX 5090** | **0.4-0.8 hr** | **8-16 hours** | **$3-6** |
| Mac M2 (MPS) | 3-5 hours      | 60-100 hours   | $0       |
| CPU          | 63 hours       | 1,260 hours    | $0       |

### Recommended Strategy

**Current Setup (Vast.AI RTX 5090):**

1. **Full training** with all layers unfrozen
2. **Batch size 64** (utilizes 32GB VRAM)
3. **20 epochs** (default, good convergence)
4. **Total time**: 8-16 hours
5. **Total cost**: $3-6

**Why RTX 5090?**

- Fastest training time (8-16 hours vs 60-100 hours on Mac)
- No thermal issues (cloud-based)
- MacBook remains free for other work
- Latest GPU architecture for optimal performance

### How Many Epochs?

- **Minimum**: 5-10 epochs (for frozen backbone)
- **Recommended**: 10-15 epochs (for frozen backbone)
- **Maximum**: 20+ epochs (diminishing returns)

**Note**: With frozen backbone, model learns faster (only classification head), so fewer epochs may be sufficient.

---

## 🐛 Troubleshooting

### Issue: Training too slow

**Solutions**:

- **Use RTX 5090 on Vast.AI** - 7-12x faster than local training
- Increase batch_size (RTX 5090 has 32GB VRAM, can use 64-128)
- Use `screen` or `tmux` to keep training running if disconnected
- Monitor GPU usage: `watch -n 1 nvidia-smi`

**Note**: The code automatically uses CUDA when available. Check training output for "Using device: cuda"

### Issue: CUDA out of memory

**Solution**: Reduce batch_size (e.g., 16 or 8)

### Issue: Model not learning

**Solution**:

- Check learning rate (try 1e-3 or 1e-5)
- Check data loading (verify images are correct)
- Check labels (verify they match folders)
- If using frozen backbone, try unfreezing last 2-3 layers

### Issue: Low AUC score

**Solution**:

- Train for more epochs
- Try different learning rate
- Check if embeddings are normalized
- Verify verification pairs are correct
- Consider unfreezing more backbone layers (slower but better)

### Issue: Training Disconnects on Vast.AI

**Problem**: Training stops when SSH connection is lost.

**Solution**: Use `screen` or `tmux` to keep training running:

```bash
# Start screen session
screen -S training

# Run training
python3 src/train_classification.py --data_dir classification_data --save_dir models

# Detach: Press Ctrl+A, then D
# Training continues even if you disconnect

# Reattach later
screen -r training

# List all screen sessions
screen -ls
```

**Alternative: Use tmux**

```bash
# Start tmux session
tmux new -s training

# Run training
python3 src/train_classification.py --data_dir classification_data --save_dir models

# Detach: Press Ctrl+B, then D
# Reattach: tmux attach -t training
```

**Note**: With `screen` or `tmux`, training continues even if you close your laptop or lose connection.

---

## ✅ Final Checklist Before Submission

- [ ] Both face verification models trained and evaluated
- [ ] ROC curves and AUC scores computed
- [ ] Anti-spoofing module integrated
- [ ] Emotion detection module integrated
- [ ] GUI functional with all features
- [ ] Project report complete (Methodology + Results sections)
- [ ] Code well-commented and organized
- [ ] Demo prepared and tested
- [ ] All requirements met according to marking scheme

---

## 🎯 Focus Areas for 80%

1. **Get classification approach working well** (easier, high ROI)
2. **Get basic metric learning working** (moderate effort)
3. **Use pre-trained models for anti-spoofing and emotion** (easy, saves time)
4. **Build simple but functional GUI** (easy, Streamlit is quick)
5. **Write clear, comprehensive report** (document as you build)

**Estimated Total Time**: 24 days (with buffer for debugging and improvements)

---

**Good luck! You've got this! 🚀**
