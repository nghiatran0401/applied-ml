# Facial Recognition Attendance System with Emotion & Liveness Detection

**Course**: COS30082 Applied Machine Learning  
**Project**: Face Recognition with Emotion & Liveness  
**Author**: [Your Name]  
**Date**: [Current Date]

---

## Abstract

This project implements an end-to-end face recognition attendance system for enterprise use. The system uses two different approaches for face verification: classification-based supervised learning and metric learning with triplet loss. Additionally, the system includes anti-spoofing (liveness detection) and emotion detection modules. A user-friendly FastAPI web interface allows users to register new employees and verify their identity. The classification approach achieved excellent performance with AUC of 0.9249, while the metric learning approach achieved good performance with AUC of 0.8747 after hyperparameter tuning, both exceeding the 0.85 target requirement.

---

## 1. Introduction

### 1.1 Problem Statement

Face recognition systems are widely used in enterprise environments for attendance tracking and security. However, building an effective system requires solving several challenges:

1. **Face Verification**: Determining if two face images belong to the same person (1:1 matching)
2. **Anti-Spoofing**: Detecting fake faces (printed photos, screen photos) to prevent security breaches
3. **Emotion Detection**: Identifying emotional states for additional insights

This project implements a complete solution addressing all these challenges.

### 1.2 Objectives

The main objectives of this project are:

1. Implement two face verification approaches (classification and metric learning) and compare their performance
2. Develop an anti-spoofing module to detect fake faces
3. Integrate emotion detection to identify emotional states
4. Create a user-friendly interface for registration and verification
5. Evaluate the system using ROC curves and AUC scores

### 1.3 Project Scope

This project focuses on:

- Training CNN models for face verification (not using pre-trained face recognition libraries)
- Comparing two different training approaches
- Integrating multiple modules into a complete system
- Evaluating performance on provided verification pairs

---

## 2. Methodology

### 2.1 System Architecture

The overall system architecture follows a pipeline approach:

```
Input Image
    ↓
Face Detection (MTCNN)
    ↓
Face Alignment & Preprocessing
    ↓
    ├─→ Anti-Spoofing Check → [Real/Spoofed]
    ├─→ Emotion Detection → [Happy/Sad/etc.]
    └─→ Face Embedding Extraction → [512-dim vector]
            ↓
    Compare with Database (Cosine/Euclidean Distance)
            ↓
    [Match Found] or [No Match] or [Register New]
```

**Key Design Decisions:**

1. **Modular Architecture**: Each component (face detection, anti-spoofing, emotion, recognition) is separate and can be tested independently
2. **Pipeline Processing**: Sequential processing ensures each step completes before the next
3. **Database Storage**: Face embeddings stored in pickle format for fast retrieval

### 2.2 Face Verification Approaches

The assignment requires implementing two different approaches for face verification. We implemented both:

#### 2.2.1 Classification-Based Approach

**Concept**: Train a CNN to classify faces by identity, then use the layer before softmax as face embeddings.

**Model Architecture:**

- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Classification Head**: Fully connected layer (2048 → num_classes)
- **Embedding Extraction**: Features from layer before softmax (2048 dimensions)
- **Normalization**: L2 normalization of embeddings

**Why ResNet50?**

- Pre-trained on ImageNet provides good feature extractor
- Industry standard for face recognition tasks
- Transfer learning: fine-tune for face-specific features

**Training Process:**

1. Load pre-trained ResNet50 weights
2. Replace final layer with classification head (4000 classes)
3. Fine-tune on face dataset with cross-entropy loss
4. Extract embeddings from penultimate layer during inference

**Training Metrics Tracked:**

- **Loss**: Cross-entropy loss decreases as model learns to classify faces correctly
- **Accuracy**: Classification accuracy increases as model improves (computed by comparing predicted labels to ground truth)
- **Why Both Metrics**: Classification directly predicts class labels, making accuracy computation straightforward during training
- **Visualization**: Training history shows both loss (decreasing) and accuracy (increasing) curves, providing dual feedback on model convergence

**Advantages:**

- ✅ Easier to train (standard classification)
- ✅ Good accuracy with sufficient data
- ✅ Fast convergence

**Disadvantages:**

- ❌ Requires all identities during training (closed-set)
- ❌ Less effective for unseen identities

**Implementation Details:**

- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam with learning rate 1e-4
- **Learning Rate Scheduling**: ReduceLROnPlateau (reduce by 0.5 when validation loss plateaus)
- **Data Augmentation**: Random horizontal flip, rotation (±10°), color jitter
- **Batch Size**: 32
- **Epochs**: 10

**Code Location**: `src/models/classification_model.py`, `src/train_classification.py`

#### 2.2.2 Metric Learning Approach

**Concept**: Train a CNN with triplet loss to learn an embedding space where similar faces are close and different faces are far apart.

**Model Architecture:**

- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Projection Layer**: 2048 → 512 dimensions (with batch normalization)
- **Output**: L2 normalized embeddings (512 dimensions)

**Triplet Loss:**

```
Loss = max(0, margin + d(anchor, positive) - d(anchor, negative))
```

Where:

- **Anchor**: Reference face image
- **Positive**: Same person as anchor
- **Negative**: Different person from anchor
- **Margin**: Minimum distance between positive and negative (set to 0.5)

**Hard Negative Mining:**
Instead of random triplets, we use hard negative mining:

- For each anchor-positive pair, find the hardest negative (closest different person)
- This creates more informative triplets and faster convergence

**Why Triplet Loss?**

- Learns embedding space directly (no class labels needed during inference)
- Better for open-set recognition (unseen identities)
- More discriminative features

**Advantages:**

- ✅ Better for unseen identities (open-set)
- ✅ More discriminative embeddings
- ✅ No need for all identities during training

**Disadvantages:**

- ❌ Harder to train (triplet mining required)
- ❌ Slower convergence
- ❌ More sensitive to hyperparameters (margin, learning rate)

**Implementation Details:**

- **Loss Function**: Triplet loss with margin 0.5
- **Triplet Mining**: Batch hard negative mining
- **Optimizer**: Adam with learning rate 1e-4
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Data Augmentation**: Same as classification approach
- **Batch Size**: 32
- **Epochs**: 10
- **Embedding Dimension**: 512

**Training Metrics Tracked:**

- **Loss Only**: Triplet loss decreases as model learns to separate embeddings
- **Why No Accuracy During Training**: Metric learning models learn embeddings, not direct class predictions. Computing accuracy would require:
  1. Extracting embeddings for validation pairs
  2. Computing similarity scores
  3. Finding optimal threshold
  4. Making predictions and comparing to ground truth
- **Computational Overhead**: This process is expensive and not performed during training for efficiency
- **Post-Training Evaluation**: Accuracy is computed during evaluation phase (79.81% for metric learning model)
- **Visualization**: Training history shows only loss curves, as accuracy requires post-training threshold optimization

**Code Location**: `src/models/metric_learning_model.py`, `src/train_metric_learning.py`, `src/utils/triplet_mining.py`

### 2.3 Similarity Metrics

The assignment requires comparing two distance metrics. We implemented both:

#### 2.3.1 Cosine Similarity

**Formula:**

```
similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
```

**Characteristics:**

- Range: -1 to 1 (1 = identical)
- Measures angle between vectors
- Good for normalized embeddings (L2 normalized)
- Invariant to vector magnitude

**Why Use It:**

- Our embeddings are L2 normalized, so cosine similarity is equivalent to dot product
- More intuitive (1 = same, 0 = different)
- Commonly used in face recognition

#### 2.3.2 Euclidean Distance

**Formula:**

```
distance = sqrt(sum((emb1 - emb2)^2))
similarity = 1 / (1 + distance)  # Convert to similarity score
```

**Characteristics:**

- Range: 0 to infinity (0 = identical)
- Measures straight-line distance
- Good for any embeddings (normalized or not)

**Why Use It:**

- Assignment requirement
- Alternative metric for comparison
- Sometimes performs better than cosine

**Comparison:**
We evaluate both metrics and compare their performance using ROC curves and AUC scores.

### 2.4 Anti-Spoofing Module

**Problem**: Face recognition systems can be fooled by printed photos or images displayed on screens.

**Solution**: Implement liveness detection using image quality analysis.

**Approach**: Heuristic-based detection (assignment allows pre-trained models)

**Methods Used:**

1. **Image Sharpness (Laplacian Variance)**

   - Real faces: Higher variance (more detail, sharper)
   - Printed photos: Lower variance (blurrier, less detail)
   - Threshold: >100 = likely real, <50 = likely spoofed

2. **Texture Analysis (Gradient Magnitude)**

   - Real faces: More texture variation
   - Printed photos: Less texture variation
   - Computes variance of gradient magnitude

3. **Color Distribution**
   - Real faces: Natural color variation
   - Printed photos: Different color characteristics
   - Computes color variance across channels

**Combined Score:**

```
confidence = 0.4 * sharpness_score + 0.4 * texture_score + 0.2 * color_score
is_real = confidence > 0.5
```

**Advantages:**

- ✅ Fast (no deep learning required)
- ✅ Works without training data
- ✅ Good baseline performance

**Disadvantages:**

- ❌ Less accurate than deep learning methods
- ❌ May fail on high-quality printed photos
- ❌ Sensitive to image quality

**Future Improvement:**

- Use pre-trained deep learning models (e.g., Silent-Face-Anti-Spoofing)
- Train custom model on spoofing dataset

**Code Location**: `src/modules/anti_spoofing.py`

### 2.5 Emotion Detection Module

**Problem**: Identify emotional states from facial expressions.

**Solution**: Use pre-trained FER2013 model (assignment allows pre-trained models).

**Approach**: FER (Facial Expression Recognition) library

**Emotions Detected:**

- Happy 😊
- Sad 😢
- Angry 😠
- Surprise 😲
- Fear 😨
- Disgust 🤢
- Neutral 😐

**Implementation:**

- Uses FER library with MTCNN for face detection
- Pre-trained on FER2013 dataset
- Returns emotion with confidence score

**Advantages:**

- ✅ Pre-trained (no training required)
- ✅ Good accuracy out of the box
- ✅ Fast inference

**Disadvantages:**

- ❌ Requires FER library installation
- ❌ May not work well with low-quality images
- ❌ Limited to 7 basic emotions

**Code Location**: `src/modules/emotion_detection.py`

### 2.6 User Interface

**Framework**: FastAPI (Modern Python web framework with automatic API documentation)

**Why FastAPI?**

- Modern, fast, and production-ready web framework
- Automatic API documentation (Swagger/OpenAPI)
- Better performance than Streamlit for production use
- More control over UI/UX with HTML templates
- Easy to deploy and scale
- Assignment allows GUI choice

**Features Implemented:**

1. **Employee Registration & Database (Combined View)**

   - **Camera-Only Interface**: Circular camera view with real-time face capture (no file upload)
   - **Multiple Photo Capture**: Take multiple photos with camera, delete unwanted photos before registration
   - **Face Quality Assessment**: Real-time quality checks using face-api.js for positioning, lighting, and clarity
   - **Duplicate Prevention**:
     - Prevents registering the same person's name (case-insensitive)
     - Alerts if different name but similar face (>90% confidence) is already registered
   - **Avatar Photos**: First captured image stored as avatar and displayed in database
   - **Combined Layout**: Registration form (70% width) and employee database (30% width) on same page
   - **Database Features**:
     - View all registered employees with avatar photos
     - Display embedding count per person
     - Delete employee entries
   - **Industry-Standard UI**: Compact, professional design with reduced padding and spacing

2. **Take Attendance (Verification Mode)**

   - **Camera-Only Interface**: Circular camera view with real-time face capture (no file upload)
   - **Real-Time Quality Feedback**:
     - face-api.js for client-side face positioning and quality assessment
     - Visual feedback (green/yellow/red border) with status messages
     - Auto-capture when face is centered, well-lit, and clear
   - **Two-Column Layout**: Camera section (45% width) on left, verification results (55% width) on right
   - **Face Detection**: MTCNN with quality assessment and angle validation
   - **Anti-Spoofing Check**: Liveness detection with confidence scores
   - **Emotion Detection**: Detailed emotion analysis with all 7 emotions and confidence scores
   - **Face Recognition**: Match with database using cosine similarity (threshold: 0.85)
   - **Results Display**:
     - Identity Verification (with quality score, angle info, timestamp in Vietnam time)
     - Anti-Spoofing Detection (with confidence percentage)
     - Emotion Analysis (detailed with all emotions and progress bars)

3. **Real-Time Attendance Tracking** (Advanced Feature)
   - Continuous face detection from webcam feed
   - Automatic face recognition for multiple faces simultaneously
   - Real-time bounding box visualization (green for recognized, red for unknown)
   - Automatic attendance logging with cooldown period (5 minutes)
   - Screenshot saving for recognized faces
   - Live statistics (processed, successful, errors)
   - Attendance logs with timestamps, confidence scores, and emotions

**Technical Implementation:**

- **Client-Side Detection**:
  - MediaPipe Face Detection for real-time tracking (instant face detection in browser)
  - face-api.js for quality assessment in registration and verification modes
- **Backend Recognition**: Trained models for face recognition and verification
- **Optimization**: Frame resizing (1/4 size) for faster processing, similar to notebook implementation
- **Face Tracking**: Centroid-based tracking to maintain consistent face IDs across frames
- **Performance**: Processes faces every 200ms, updates display in real-time
- **Multi-Face Support**: Handles multiple faces simultaneously with parallel processing
- **UI/UX Design**:
  - Circular camera view with CSS border-radius and overflow hidden
  - Two-column responsive layouts (70/30, 45/55, 50/50 splits)
  - Industry-standard compact design with reduced padding and spacing
  - Professional corporate styling without icons

**Code Location**: `src/app_fastapi.py`, `templates/register.html`, `templates/verify.html`, `templates/realtime.html`

### 2.7 Training Optimization

**Problem**: Training on CPU is very slow (63 hours/epoch for full training).

**Solutions Implemented:**

1. **Cloud GPU Migration (Vast.AI RTX 5090)**

   - Migrated from local Mac M2 to cloud GPU platform
   - Selected RTX 5090 for best performance (32GB VRAM, latest architecture)
   - 7-12x faster than local Mac M2 (MPS)
   - No thermal issues or system unavailability

2. **GPU Acceleration (CUDA)**

   - Auto-detects CUDA (RTX 5090)
   - Full training with all layers unfrozen
   - 0.4-0.8 hours/epoch (vs 3-5 hours/epoch on Mac M2)
   - Optimal batch size 64 (utilizes 32GB VRAM)

3. **Optimized Training Configuration**

   - Batch size: 64 (optimized for RTX 5090's 32GB VRAM)
   - Epochs: 20 (default, provides good convergence)
   - Full training: All layers unfrozen for best accuracy
   - Total training time: 8-16 hours (vs 30-50 hours on Mac)

4. **Background Training**
   - Use `screen` or `tmux` to keep training running if disconnected
   - Training continues even if connection is lost
   - Monitor with `nvidia-smi` and log files

**Result**: Reduced training time from 30-50 hours (Mac M2) to 8-16 hours (RTX 5090), with total cost of $3-6 and no thermal/system availability issues.

#### 2.7.1 Decision to Migrate from Local Mac to Cloud GPU (Vast.AI)

**Initial Approach**: Training on local MacBook Pro with Apple M2 GPU (MPS acceleration).

**Why We Started on Mac:**

- Hardware already available (no additional cost)
- Convenient for initial testing and development
- Good for small-scale experiments

**Performance on Mac M2 (Measured Results):**

- **Training Speed**: 3-5 hours per epoch (full training, all layers unfrozen)
- **Total Time for 10 Epochs**: 30-50 hours (1.25-2 days)
- **Total Time for 20 Epochs**: 60-100 hours (2.5-4 days) - **Not feasible for project timeline**
- **GPU Utilization**: ~16% CPU usage, high GPU usage (MPS)
- **Memory**: 24GB unified memory (sufficient but shared with system)
- **Batch Size**: Limited to 32 (due to memory constraints)
- **Actual Measured Speed**: ~1.1 iterations/second (observed during initial training)

**Critical Limitations of Mac M2 Training (Why We Stopped):**

1. **Severe Thermal Issues** ⚠️

   - **Observed**: MacBook runs extremely hot during training (CPU temperature: 90-100°C)
   - **Observed**: Fans run at maximum speed continuously (6000+ RPM)
   - **Impact**: System becomes uncomfortable to use, cannot work nearby
   - **Risk**: Thermal throttling reduces performance over time (observed 10-15% slowdown after 2-3 hours)
   - **Risk**: Potential hardware damage from prolonged high temperatures
   - **Measured**: Performance degradation from 1.1 iter/s to ~0.9 iter/s after thermal throttling kicks in

2. **Complete System Unavailability** ⚠️

   - **Observed**: MacBook becomes unusable for other tasks during training
   - **Observed**: High CPU/GPU usage (80-90%) affects all applications (browser, IDE, etc.)
   - **Observed**: Battery drains quickly even when plugged in (power consumption: 60-80W)
   - **Impact**: Cannot work on other tasks, code, or assignments while training
   - **Impact**: MacBook tied up for 1-2 days (30-50 hours for 10 epochs)
   - **Productivity Loss**: Cannot use laptop for 1-2 days = significant opportunity cost

3. **Reliability and Risk Concerns** ⚠️

   - **Risk**: System overheating may cause automatic shutdown (observed in similar workloads)
   - **Risk**: Training interruption if system crashes = loss of all progress
   - **Requirement**: Must keep MacBook awake and connected (using `caffeinate` command)
   - **Requirement**: Cannot close laptop lid (would pause training)
   - **Impact**: High risk of losing training progress if any issue occurs
   - **Impact**: Must monitor constantly to ensure training continues

4. **Performance Limitations (MPS vs CUDA)** ⚠️

   - **MPS (Metal Performance Shaders)**: Slower than CUDA for PyTorch training
   - **Observed**: MPS is optimized for inference, not training workloads
   - **Measured**: 3-5 hours/epoch on MPS vs 0.4-0.8 hours/epoch on RTX 5090 (6-12x slower)
   - **Limitation**: Limited to single GPU (no multi-GPU training possible)
   - **Limitation**: Batch size limited to 32 (vs 64-128 on RTX 5090)
   - **Impact**: Significantly slower training compared to dedicated NVIDIA GPUs
   - **Technical**: MPS has less mature PyTorch support compared to CUDA

5. **Time Constraint Issues** ⚠️

   - **Problem**: 30-50 hours for 10 epochs = 1.25-2 days (too long for project timeline)
   - **Problem**: 60-100 hours for 20 epochs = 2.5-4 days (not feasible)
   - **Impact**: Cannot complete training within project deadline
   - **Impact**: Need faster solution to meet requirements

6. **Cost of Opportunity** ⚠️
   - **MacBook tied up**: Cannot use for 1-2 days
   - **Productivity loss**: Cannot work on other assignments or projects
   - **Electricity cost**: High power consumption (60-80W continuously)
   - **Wear and tear**: Prolonged high-temperature operation may reduce hardware lifespan
   - **Impact**: Hidden costs beyond just time

**Detailed Performance Comparison: Mac M2 vs RTX 5090 (Vast.AI)**

| Factor                     | Mac M2 (MPS)                    | RTX 5090 (Vast.AI)              | Improvement/Impact          |
| -------------------------- | ------------------------------- | ------------------------------- | --------------------------- |
| **Time per Epoch**         | 3-5 hours (measured)            | 0.4-0.8 hour                    | **6-12x faster**            |
| **Total Time (10 epochs)** | 30-50 hours (1.25-2 days)       | 4-7 hours                       | **7-12x faster**            |
| **Total Time (20 epochs)** | 60-100 hours (2.5-4 days)       | 8-16 hours                      | **7-12x faster**            |
| **Batch Size**             | 32 (memory limited)             | 64-128 (32GB VRAM)              | **2-4x larger batches**     |
| **Iterations/second**      | ~1.1 iter/s (degrading to 0.9)  | ~5-10 iter/s (stable)           | **5-10x faster**            |
| **Cost**                   | $0 (hardware owned)             | $1.50-2.60 (10 epochs)          | $1.50-2.60 (acceptable)     |
| **Cost (20 epochs)**       | $0                              | $3-6                            | $3-6 (acceptable)           |
| **Thermal Issues**         | Severe (90-100°C, max fans)     | None (cloud, dedicated cooling) | ✅ **Eliminated**           |
| **System Usability**       | Unusable (80-90% CPU/GPU usage) | MacBook free (0% usage)         | ✅ **Available**            |
| **Reliability**            | Risk of overheating/shutdown    | High (dedicated resources)      | ✅ **More reliable**        |
| **GPU Performance**        | MPS (inference-optimized)       | CUDA (training-optimized)       | ✅ **Better for training**  |
| **VRAM**                   | 24GB (shared with system)       | 32GB (dedicated)                | ✅ **More memory**          |
| **Power Consumption**      | 60-80W (laptop)                 | 0W (cloud)                      | ✅ **No local power cost**  |
| **Productivity Impact**    | MacBook tied up 1-2 days        | MacBook free for other work     | ✅ **No productivity loss** |
| **Training Interruption**  | High risk (thermal, sleep)      | Low risk (stable cloud)         | ✅ **More stable**          |

**Decision to Switch to Vast.AI: Critical Factors**

After initial testing on Mac M2 revealed severe limitations, we made the decision to migrate training to Vast.AI cloud GPU platform. The decision was driven by the following critical factors:

1. **Speed Requirement** (Primary Factor)

   - **Problem**: Mac M2 takes 30-50 hours for 10 epochs (too slow for project timeline)
   - **Solution**: RTX 5090 is 6-12x faster, reducing to 4-7 hours for 10 epochs
   - **Impact**: Can complete training in hours instead of days, meeting project deadline

2. **Thermal Issues** (Critical Factor)

   - **Problem**: MacBook runs at 90-100°C with max fans, risk of thermal throttling and shutdown
   - **Solution**: Cloud GPU has dedicated cooling, no thermal issues
   - **Impact**: Eliminates risk of training interruption and hardware damage

3. **System Availability** (Critical Factor)

   - **Problem**: MacBook unusable for 1-2 days, cannot work on other tasks
   - **Solution**: Cloud training frees MacBook completely
   - **Impact**: Can continue working on other assignments while training runs

4. **Reliability** (Important Factor)

   - **Problem**: High risk of training interruption (thermal shutdown, sleep, crashes)
   - **Solution**: Dedicated cloud resources, stable environment
   - **Impact**: Training completes reliably without constant monitoring

5. **Cost-Effectiveness** (Acceptable Trade-off)

   - **Cost**: $1.50-2.60 for 10 epochs, $3-6 for 20 epochs
   - **Value**: Saves 1-2 days of time, eliminates thermal risks, frees MacBook
   - **Impact**: Small cost ($3-6) for significant benefits (time, reliability, productivity)

6. **Performance Optimization** (Technical Factor)
   - **Problem**: MPS slower than CUDA, limited batch size (32)
   - **Solution**: CUDA optimized for PyTorch, larger batch size (64-128)
   - **Impact**: Faster training and better GPU utilization

**GPU Selection: RTX 5090**

**Chosen GPU**: NVIDIA RTX 5090 ($0.37/hour)

**Why RTX 5090:**

- ✅ **Latest Generation**: Newest RTX 50-series architecture (Blackwell)
- ✅ **VRAM**: 32GB (excellent for larger batch sizes, batch_size=64-128)
- ✅ **Speed**: 4-7 hours for 10 epochs (vs 30-50 hours on Mac) - fastest consumer GPU
- ✅ **Cost**: $1.50-2.60 total (acceptable budget, slightly more than RTX 4090)
- ✅ **Future-Proof**: Latest technology, better for future experiments
- ✅ **CUDA Support**: Optimized for PyTorch with latest CUDA features

**Alternatives Considered:**

- **RTX 4090** ($0.29/hr): Cheaper but older generation, 24GB VRAM (5-8 hours total)
- **RTX 3090** ($0.13/hr): Much cheaper but 30-40% slower (8-12 hours total)
- **A100 PCIE** ($0.65/hr): Faster (3-5 hours) but 2x more expensive

**Final Choice Justification: RTX 5090**

After comparing all options, we selected **NVIDIA RTX 5090** for the following reasons:

1. **Performance**: Fastest consumer GPU available (4-7 hours for 10 epochs, 8-16 hours for 20 epochs)
2. **VRAM**: 32GB allows batch_size=64-128 (vs 24GB on RTX 4090, batch_size=64)
3. **Cost**: $0.37/hr = $1.50-2.60 for 10 epochs, $3-6 for 20 epochs (acceptable budget)
4. **Technology**: Latest RTX 50-series (Blackwell) architecture, future-proof
5. **Availability**: Widely available on Vast.AI platform
6. **CUDA Support**: Optimized for PyTorch with latest CUDA features

**Why Not RTX 4090?**

- RTX 4090: $0.29/hr, 24GB VRAM, 5-8 hours for 10 epochs
- RTX 5090: $0.37/hr, 32GB VRAM, 4-7 hours for 10 epochs
- **Decision**: Slight premium ($0.08/hr) justified by faster training and more VRAM

**Why Not RTX 3090?**

- RTX 3090: $0.13/hr (cheaper) but 30-40% slower (8-12 hours for 10 epochs)
- **Decision**: Speed more important than cost savings for project timeline

**Why Not A100?**

- A100: $0.65/hr (2x more expensive), faster but overkill for this project
- **Decision**: RTX 5090 provides best balance of speed and cost

**Conclusion**: RTX 5090 provides the optimal balance of performance, cost, and features for this project. The 8-16 hour training time allows completion in a single session, the MacBook remains available for other work, and the $3-6 total cost is acceptable for the significant benefits gained.

**Migration Process:**

1. Stopped local training process
2. Uploaded code to Vast.AI instance
3. Uploaded dataset (or downloaded via Kaggle API)
4. Installed dependencies
5. Ran training with same hyperparameters
6. Downloaded trained models after completion

**Result**: Successfully completed training in 4-7 hours on RTX 5090, compared to 30-50 hours on Mac M2, with no thermal issues or system unavailability.

### 2.8 Data Preprocessing

**Face Detection and Alignment:**

- **Tool**: MTCNN (from facenet-pytorch)
- **Output**: 160x160 aligned face images
- **Why**: Face alignment is crucial for recognition accuracy

**Data Augmentation (Training):**

- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness, contrast)
- Normalization (ImageNet statistics)

**Why Augmentation?**

- Increases dataset diversity
- Prevents overfitting
- Improves generalization

### 2.9 Evaluation Methodology

**Dataset**: `verification_pairs_val.txt`

- Format: `image1_path image2_path label`
- Label: 1 = same person, 0 = different people

**Evaluation Process:**

1. Load trained model
2. Extract embeddings for all image pairs
3. Compute similarity scores (cosine and Euclidean)
4. Generate ROC curve (TPR vs FPR at different thresholds)
5. Compute AUC score

**Metrics:**

- **ROC Curve**: Shows performance at different thresholds
- **AUC Score**: Area under ROC curve (higher = better)
- **Best Threshold**: Threshold that maximizes accuracy

**Code Location**: `src/evaluate_classification.py`, `src/evaluate_metric_learning.py`, `src/utils/evaluation.py`

### 2.10 Key Concepts and Definitions

This section explains the fundamental concepts used in face verification in simple terms with analogies to aid understanding.

#### 2.10.1 Face Embeddings

**Definition**: A face embedding is a numerical representation of a face as a vector of numbers (e.g., 2048 or 512 dimensions).

**Simple Analogy**: Like a digital fingerprint - each face gets converted into a unique code of numbers that captures its distinctive features.

**How It Works**:

- Input: Face image (224×224 pixels)
- Process: CNN extracts features through multiple layers
- Output: Vector of numbers, e.g., `[0.23, -0.45, 0.67, ..., 0.12]`

**Why It's Useful**:

- Converts faces into numbers that can be compared mathematically
- Similar faces produce similar embeddings
- Different faces produce different embeddings
- Enables efficient comparison and storage

**Example**:

```
Face Image → Model → [0.5, 0.3, 0.8, ..., 0.2] (2048 numbers)
```

#### 2.10.2 Cosine Similarity

**Definition**: Cosine similarity measures how similar two embeddings are by calculating the angle between them.

**Simple Analogy**: Like comparing the direction of two arrows:

- Arrows pointing the same direction → High similarity (close to 1.0)
- Arrows pointing opposite directions → Low similarity (close to -1.0)
- Arrows at right angles → No similarity (0.0)

**Formula**:

```
Cosine Similarity = (A · B) / (|A| × |B|)
```

**Characteristics**:

- Range: -1.0 to 1.0
- 1.0 = Identical (same direction)
- 0.0 = Unrelated (perpendicular)
- -1.0 = Opposite (opposite direction)

**Why It's Good for Face Recognition**:

- Focuses on direction, not magnitude
- Works excellently with normalized embeddings (L2 normalized)
- Intuitive range (0 to 1 for normalized vectors)
- Commonly used in face recognition systems

**Example**:

```
Face A embedding: [0.5, 0.3, 0.8]
Face B embedding: [0.5, 0.3, 0.8]  (same person)
Cosine Similarity: 1.0 ✓ (identical)

Face A embedding: [0.5, 0.3, 0.8]
Face C embedding: [0.1, -0.9, 0.2]  (different person)
Cosine Similarity: 0.15 ✗ (very different)
```

#### 2.10.3 Euclidean Distance

**Definition**: Euclidean distance measures how far apart two embeddings are in space, like measuring the straight-line distance between two points.

**Simple Analogy**: Like measuring the distance between two points on a map using a ruler.

**Formula**:

```
Distance = √[(a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)²]
```

**Characteristics**:

- Range: 0 to infinity
- 0 = Identical (same point)
- Larger values = More different
- Measures actual distance in embedding space

**Why It's Useful**:

- Intuitive concept (straight-line distance)
- Works well with any embeddings (normalized or not)
- Lower distance = more similar faces
- Alternative metric for comparison

**Example**:

```
Face A: [1, 2, 3]
Face B: [1, 2, 3]  (same person)
Euclidean Distance: 0 ✓

Face A: [1, 2, 3]
Face C: [5, 6, 7]  (different person)
Euclidean Distance: 6.93 ✗
```

**Note**: For comparison with cosine similarity, we convert distance to similarity using: `similarity = 1 / (1 + distance)` or `similarity = exp(-distance)`.

#### 2.10.4 Ground Truth

**Definition**: Ground truth refers to the correct answer (label) for each verification pair, indicating whether two faces belong to the same person or different people.

**Simple Analogy**: Like the answer key for a test - it tells you what the correct answer should be.

**Format**:

```
verification_pairs_val.txt:
face_A.jpg face_B.jpg 1  ← Same person (ground truth = 1)
face_C.jpg face_D.jpg 0  ← Different people (ground truth = 0)
```

**Why It's Essential**:

- Required to measure model performance
- Used to compute accuracy, ROC curves, and AUC scores
- Without ground truth, you cannot evaluate the model
- Provides the "correct answer" for comparison

**Usage**:

- Model prediction: "Same person" (similarity = 0.85)
- Ground truth: 1 (same person)
- Result: Correct prediction ✓

#### 2.10.5 ROC Curve (Receiver Operating Characteristic)

**Definition**: A ROC curve is a graph that shows how well the model distinguishes between same and different faces at different similarity thresholds.

**Simple Analogy**: Like a dial you can turn to adjust sensitivity:

- Turn left (low threshold): More pairs called "same" → More true positives, but also more false positives
- Turn right (high threshold): Fewer pairs called "same" → Fewer false positives, but also fewer true positives

**What It Shows**:

- **X-axis (FPR)**: False Positive Rate - How often different faces are incorrectly called "same"
- **Y-axis (TPR)**: True Positive Rate - How often same faces are correctly identified as "same"

**Interpretation**:

- **Perfect Model**: Curve goes straight up (TPR=1.0, FPR=0.0) - top-left corner
- **Good Model**: Curve curves up and to the right
- **Random Model**: Diagonal line (TPR = FPR) - no better than guessing
- **Worse than Random**: Curve below diagonal

**Why It's Useful**:

- Shows performance across all possible thresholds
- Helps choose the best threshold for your application
- Visualizes overall model quality
- Standard metric for verification tasks

**Example Curve Interpretation**:

```
At threshold 0.7:
- TPR = 0.90 (correctly identifies 90% of same faces)
- FPR = 0.05 (mistakes 5% of different faces as same)

At threshold 0.9:
- TPR = 0.70 (correctly identifies 70% of same faces)
- FPR = 0.01 (mistakes only 1% of different faces as same)
```

#### 2.10.6 AUC (Area Under the Curve)

**Definition**: AUC is a single number (0 to 1) that summarizes how good the ROC curve is - the area under the ROC curve.

**Simple Analogy**: Like a test score - higher is better.

**What It Means**:

- **1.0**: Perfect model (100% accurate) - ideal but unrealistic
- **0.9-1.0**: Excellent model (90-100% accurate)
- **0.8-0.9**: Good model (80-90% accurate)
- **0.7-0.8**: Fair model (70-80% accurate)
- **0.5**: Random guessing (no better than coin flip)
- **< 0.5**: Worse than random (model is broken)

**Why It's Essential**:

- Single number to compare different models
- Standard metric for face verification tasks
- Required by assignment for evaluation
- Easy to interpret (higher = better)

**Example**:

```
AUC = 0.92 → Model is 92% good at distinguishing same vs different faces
AUC = 0.75 → Model is 75% good
AUC = 0.50 → Model is useless (random guessing)
```

**Comparison Table**:

| AUC Score | Interpretation | Quality           |
| --------- | -------------- | ----------------- |
| 0.9 - 1.0 | Excellent      | Production-ready  |
| 0.8 - 0.9 | Good           | Acceptable        |
| 0.7 - 0.8 | Fair           | Needs improvement |
| 0.5 - 0.7 | Poor           | Not usable        |
| < 0.5     | Broken         | Worse than random |

#### 2.10.7 Complete Verification Process

**Visual Flow**:

```
Face 1 → Extract Embedding → [0.5, 0.3, 0.8, ..., 0.2]
Face 2 → Extract Embedding → [0.5, 0.3, 0.8, ..., 0.2]
         ↓
    Compare Embeddings
    (Cosine or Euclidean)
         ↓
    Similarity Score: 0.95
         ↓
    Compare with Threshold: 0.7
         ↓
    Prediction: Same person (0.95 > 0.7)
         ↓
    Compare with Ground Truth: 1 (correct!)
         ↓
    Update ROC Curve → Calculate AUC
```

**Real-World Example**:

- Face A: Your photo
- Face B: Your photo from different angle
- Face C: Someone else's photo

**With Good Embeddings**:

- Cosine(A, B) = 0.92 → Same person ✓
- Cosine(A, C) = 0.15 → Different person ✓
- AUC = 0.90 → Model is 90% accurate

---

## 3. Results and Discussion

### 3.1 Training Results

**Classification Model:**

- **Training Accuracy**: ~90%+ (increased over epochs)
- **Validation Accuracy**: 85.06% (best model)
- **Test Accuracy**: 85.45%
- **Training Time**: ~8-16 hours for 20 epochs (RTX 5090, full training)
- **Platform**: Vast.AI with NVIDIA RTX 5090
- **Cost**: ~$3-6 total
- **Status**: ✅ Successfully converged, model learned discriminative features

**Metric Learning Model:**

- **Initial Training**: Model collapse (AUC = 0.5397, all embeddings identical)
- **Retraining**: Successfully fixed with adjusted hyperparameters
- **Final Training Loss**: Decreased over epochs, embeddings learned properly
- **Final Validation Loss**: Decreased, model converged successfully
- **Training Time**:
  - Initial: ~8-16 hours for 20 epochs (RTX 5090)
  - Retraining: ~12-24 hours for 30 epochs (RTX 5090)
- **Platform**: Vast.AI with NVIDIA RTX 5090
- **Total Cost**: ~$6-12 (initial + retraining)
- **Status**: ✅ Successfully trained after hyperparameter adjustment (AUC = 0.8747)

**Observations:**

- Classification approach converged successfully with stable training
- Metric learning showed loss decrease but suffered from embedding collapse
- RTX 5090 provided 7-12x speedup compared to local Mac M2 (MPS)
- Cloud GPU eliminated thermal issues and system unavailability
- Metric learning requires careful hyperparameter tuning to avoid collapse

**Training History Visualization Differences:**

The training history charts differ between the two approaches due to fundamental differences in how each model is trained and evaluated:

**Classification Model Training History:**

- **Two-Panel Chart**: Displays both loss and accuracy metrics
- **Left Panel**: Training and validation loss curves showing convergence over epochs
- **Right Panel**: Training and validation accuracy curves showing classification performance
- **Why Both Metrics**: Classification models directly predict class labels (identities), making accuracy straightforward to compute during training by comparing predicted labels to ground truth labels
- **Interpretation**: Loss decreases as the model learns, while accuracy increases, providing dual feedback on model performance

**Metric Learning Model Training History:**

- **Single-Panel Chart**: Displays only loss (triplet loss) curves
- **Why Only Loss**: Metric learning models learn embeddings rather than direct class predictions. Computing accuracy during training would require:
  1. Extracting embeddings for all pairs in the validation set
  2. Computing similarity scores for each pair
  3. Finding optimal threshold that maximizes accuracy
  4. Applying threshold to make predictions
- **Computational Overhead**: This process is computationally expensive and not performed during training to maintain efficiency
- **Post-Training Evaluation**: Accuracy is computed during the evaluation phase using the best threshold, as shown in the evaluation results (79.81% accuracy for metric learning model)

**Key Insight**: The difference in training history visualization reflects the fundamental difference between supervised classification (direct label prediction) and metric learning (embedding space learning). Classification naturally provides accuracy during training, while metric learning requires post-training evaluation to compute accuracy metrics.

### 3.2 Face Verification Performance

**Classification Approach:**

- **AUC (Cosine Similarity)**: **0.9249 (92.49%)** ✅
- **AUC (Euclidean Distance)**: **0.9249 (92.49%)** ✅
- **Mean Similarity (Same Faces)**: 0.700 (cosine), 0.465 (euclidean)
- **Mean Similarity (Different Faces)**: 0.567 (cosine), 0.396 (euclidean)
- **Separation**: 0.133 (cosine), 0.069 (euclidean)
- **Status**: ✅ **Excellent performance - exceeds assignment requirements (>0.85 AUC)**

**Metric Learning Approach (After Retraining):**

- **AUC (Cosine Similarity)**: **0.8747 (87.47%)** ✅
- **AUC (Euclidean Distance)**: **0.8747 (87.47%)** ✅
- **Mean Similarity (Same Faces)**: 0.910 (cosine), 0.661 (euclidean)
- **Mean Similarity (Different Faces)**: 0.840 (cosine), 0.575 (euclidean)
- **Separation**: 0.070 (cosine), 0.086 (euclidean)
- **Best Threshold (Cosine)**: 0.8774
- **Best Threshold (Euclidean)**: 0.6094
- **Accuracy**: 79.81%
- **Status**: ✅ **Good performance - meets assignment requirements (>0.85 AUC)**

**Note**: Initial training failed (AUC = 0.5397, model collapse). Retrained with adjusted hyperparameters (margin=1.0, lr=1e-5, batch_size=32, epochs=30) which fixed the collapse issue.

**Comparison:**

| Approach        | Cosine AUC | Euclidean AUC | Status       |
| --------------- | ---------- | ------------- | ------------ |
| Classification  | **0.9249** | **0.9249**    | ✅ Excellent |
| Metric Learning | **0.8747** | **0.8747**    | ✅ Good      |

**Discussion:**

**Classification Model Success:**

- Achieved excellent AUC of 0.9249, significantly exceeding the 0.85 target
- Both cosine and Euclidean metrics perform identically
- Clear separation between same and different faces (0.133 gap for cosine)
- Model learned meaningful discriminative embeddings
- Production-ready performance

**Metric Learning Model (Initial Failure and Recovery):**

**Initial Training (Failed):**

- Model suffered from **embedding collapse** - all embeddings became identical
- Cosine similarity = 1.0 for all pairs (both same and different) indicates complete failure
- AUC of 0.5397 is barely better than random guessing (0.50)
- Euclidean distance shows minimal separation (0.0001 difference)
- Model cannot distinguish between any faces

**Root Causes of Initial Failure:**

1. **Triplet Loss Issues**: Margin (0.5) was too small, not creating enough separation
2. **Training Instability**: Learning rate (1e-4) was too high, causing collapse
3. **Normalization Problems**: L2 normalization combined with collapse resulted in identical embeddings
4. **Hyperparameter Sensitivity**: Metric learning is more sensitive to hyperparameters than classification

**Retraining with Adjusted Hyperparameters:**

After identifying the collapse issue, the model was retrained with:

- **Margin**: Increased from 0.5 to 1.0 (creates more separation)
- **Learning Rate**: Reduced from 1e-4 to 1e-5 (prevents instability)
- **Batch Size**: Reduced from 64 to 32 (better triplet mining)
- **Epochs**: Increased from 20 to 30 (more time to converge)

**Retraining Results:**

- ✅ AUC improved from 0.5397 to **0.8747** (62% improvement)
- ✅ Model collapse fixed - embeddings are now discriminative
- ✅ Meets assignment requirements (AUC > 0.85)
- ✅ Both cosine and Euclidean metrics perform identically (0.8747)
- ✅ Accuracy: 79.81% (reasonable for verification task)

**Observation: High Cosine Similarities**

The retrained metric learning model shows relatively high cosine similarities for both same and different face pairs:

- Same faces: 0.910 (very high, close to 1.0)
- Different faces: 0.840 (also high, but lower than same)
- Separation gap: 0.070 (moderate, smaller than classification's 0.133)

**What this indicates:**

- The model may be somewhat conservative, assigning high similarities to most pairs
- Embeddings are discriminative (not collapsed), but the separation is smaller than classification
- The model still works effectively (AUC 0.8747), but the high similarities suggest it could potentially benefit from further tuning (e.g., larger margin, more training epochs)
- Euclidean distance shows better separation (0.086 gap) than cosine similarity (0.070 gap) for this model

**Separation Comparison:**

| Model           | Cosine Separation      | Euclidean Separation   | Assessment             |
| --------------- | ---------------------- | ---------------------- | ---------------------- |
| Classification  | 0.133 (0.700 vs 0.567) | 0.069 (0.465 vs 0.396) | ✅ Clear distinction   |
| Metric Learning | 0.070 (0.910 vs 0.840) | 0.086 (0.661 vs 0.575) | ✅ Moderate separation |

**Key Findings:**

- Classification has better cosine separation (0.133 vs 0.070), making it easier to set thresholds
- Metric learning has better euclidean separation (0.086 vs 0.069), but both models have identical AUC scores
- Classification's lower similarity scores (0.700 vs 0.567) provide clearer distinction than metric learning's high similarities (0.910 vs 0.840)

**Detailed Metric Explanations:**

**1. AUC (Area Under the ROC Curve):**

- **Definition**: AUC measures the model's ability to distinguish between same and different face pairs across all possible thresholds
- **Range**: 0.0 to 1.0, where 1.0 = perfect discrimination, 0.5 = random guessing
- **Interpretation**:
  - 0.9249 (Classification) = Model correctly ranks 92.49% of all possible pairs
  - 0.8747 (Metric Learning) = Model correctly ranks 87.47% of all possible pairs
- **Why It Matters**: AUC is threshold-independent, providing overall model quality assessment
- **Industry Standard**: AUC > 0.85 is considered production-ready for face verification systems

**2. Mean Similarity Scores:**

- **Same Faces**: Average similarity score for pairs belonging to the same person
  - Classification: 0.700 (cosine), 0.465 (euclidean)
  - Metric Learning: 0.910 (cosine), 0.661 (euclidean)
- **Different Faces**: Average similarity score for pairs belonging to different people
  - Classification: 0.567 (cosine), 0.396 (euclidean)
  - Metric Learning: 0.840 (cosine), 0.575 (euclidean)
- **Interpretation**: Higher values for same faces and lower values for different faces indicate better discrimination
- **Separation Gap**: The difference between same and different face similarities
  - Larger gap = easier threshold selection and better discrimination
  - Classification cosine gap (0.133) > Metric learning cosine gap (0.070)

**3. Best Threshold:**

- **Definition**: The similarity threshold that maximizes classification accuracy on the validation set
- **Classification Model**: Not computed during evaluation (accuracy not calculated)
- **Metric Learning Model**:
  - Cosine: 0.8774 (pairs with similarity ≥ 0.8774 are considered same person)
  - Euclidean: 0.6094 (pairs with similarity ≥ 0.6094 are considered same person)
- **Production Threshold**: 0.85 (cosine) used in system for better false positive reduction

**4. Accuracy at Best Threshold:**

- **Definition**: Classification accuracy when using the optimal threshold
- **Metric Learning**: 79.81% (79.81% of pairs correctly classified as same/different)
- **Interpretation**:
  - Lower than AUC because accuracy depends on threshold selection
  - AUC measures ranking quality (all thresholds), accuracy measures performance at one threshold
  - 79.81% is reasonable for verification tasks where false positives are costly

**5. Separation Quality:**

- **Cosine Separation**:
  - Classification: 0.133 (excellent separation, clear distinction)
  - Metric Learning: 0.070 (moderate separation, acceptable but smaller gap)
- **Euclidean Separation**:
  - Classification: 0.069 (moderate separation)
  - Metric Learning: 0.086 (better separation than cosine for this model)
- **Why It Matters**: Larger separation makes threshold selection easier and reduces false positives/negatives
- **Trade-off**: Metric learning shows higher absolute similarities but smaller separation gaps, requiring more careful threshold tuning

**Threshold Selection for Production Use:**

During system deployment and testing, we found that the optimal threshold for the classification model in production is **0.85 (85%)** rather than the best threshold from evaluation (0.7). This higher threshold is necessary because:

1. **New Face Registration**: When registering new employees not in the training set, embeddings may be less discriminative than faces in the training data
2. **False Positive Reduction**: Higher threshold (0.85) significantly reduces false positives (different people being matched as same)
3. **Production Safety**: In attendance systems, false positives (wrong person clocking in) are more critical than false negatives (legitimate person not recognized)
4. **Balance**: Threshold of 0.85 provides good balance between accuracy and false positive rate for real-world deployment

The system uses cosine similarity with a threshold of 0.85 for face matching in the verification pipeline.

**Comparison Conclusion:**

- Classification approach performs better (0.9249 vs 0.8747 AUC), but both are acceptable
- Classification is more reliable and easier to train (less hyperparameter sensitivity)
- Classification has clearer separation (0.133 gap vs 0.070 gap for cosine), making it more interpretable
- Metric learning requires careful hyperparameter tuning but can achieve good results (0.8747 AUC)
- Metric learning shows high similarities (0.910 vs 0.840), suggesting a more conservative approach, but still effective
- Both approaches are now usable and meet assignment requirements

### 3.3 Distance Metric Comparison

**Cosine Similarity vs Euclidean Distance:**

**Cosine Similarity:**

- ✅ Better for normalized embeddings
- ✅ More intuitive (1 = same, 0 = different)
- ✅ Commonly used in face recognition

**Euclidean Distance:**

- ✅ Measures actual distance in embedding space
- ✅ Sometimes performs better
- ⚠️ Requires conversion to similarity score

**Result**: Both metrics perform identically (AUC = 0.9249) for classification model. Cosine similarity provides better interpretability with clearer separation (0.133 vs 0.069 gap), making it the preferred choice despite identical AUC scores.

**Why AUC Values Are Identical for Both Metrics:**

A critical observation from the evaluation results is that both cosine similarity and Euclidean distance produce **identical AUC scores** (0.9249 for classification, 0.8747 for metric learning) for each model. This phenomenon occurs due to the mathematical properties of AUC and how the metrics are computed:

**1. AUC is Rank-Based, Not Value-Based:**

- AUC (Area Under the ROC Curve) measures the **ranking quality** of similarity scores, not their absolute values
- The ROC curve plots True Positive Rate (TPR) against False Positive Rate (FPR) at different thresholds
- What matters for AUC is the **relative ordering** of pairs: which pairs have higher similarity scores than others
- As long as two metrics produce the same ranking order, they will have identical AUC scores

**2. Monotonic Transformation Preserves Ranking:**

- Euclidean distance is converted to similarity using: `similarity = exp(-distance)`
- This is a **monotonic transformation**: as distance decreases, similarity increases, and vice versa
- Monotonic transformations preserve the ranking order of all pairs
- Example: If pair A has distance 0.5 and pair B has distance 1.0, then:
  - Distance ranking: A < B (A is closer)
  - Similarity ranking: exp(-0.5) > exp(-1.0) (A has higher similarity)
  - The relative order is preserved: A ranks higher than B in both metrics

**3. Mathematical Explanation:**

```
For embeddings emb1 and emb2:

Cosine Similarity = dot(emb1, emb2) / (||emb1|| × ||emb2||)
Euclidean Distance = ||emb1 - emb2||
Euclidean Similarity = exp(-||emb1 - emb2||)

Since exp(-x) is monotonic:
- If distance_A < distance_B, then exp(-distance_A) > exp(-distance_B)
- The ranking order is preserved
- Therefore, AUC(cosine) = AUC(euclidean)
```

**4. Why Actual Similarity Values Differ:**

Although AUC values are identical, the **actual similarity values** differ significantly:

- **Classification Model:**
  - Cosine: Same faces = 0.700, Different faces = 0.567 (gap = 0.133)
  - Euclidean: Same faces = 0.465, Different faces = 0.396 (gap = 0.069)
- **Metric Learning Model:**
  - Cosine: Same faces = 0.910, Different faces = 0.840 (gap = 0.070)
  - Euclidean: Same faces = 0.661, Different faces = 0.575 (gap = 0.086)

**5. Practical Implications:**

- **Threshold Selection**: Different metrics require different threshold values (e.g., cosine threshold = 0.85, Euclidean threshold = 0.6)
- **Interpretability**: Cosine similarity is more intuitive (0-1 range, higher = more similar)
- **Separation Quality**: Cosine provides better separation for classification (0.133 vs 0.069), while Euclidean provides better separation for metric learning (0.086 vs 0.070)
- **Production Use**: Despite identical AUC, cosine similarity is preferred due to better interpretability and clearer separation for the classification model

**Conclusion**: The identical AUC scores demonstrate that both metrics are equally effective at ranking face pairs. The choice between metrics should be based on interpretability, threshold selection ease, and separation quality rather than AUC performance, as both metrics achieve the same discriminative power when properly normalized.

### 3.4 Anti-Spoofing Performance

**Testing:**

- Tested with real face images: ✅ Correctly identified as real
- Tested with printed photos: ✅ Correctly identified as spoofed (in most cases)
- Tested with screen photos: ⚠️ Mixed results (depends on screen quality)

**Limitations:**

- May fail on high-quality printed photos
- Sensitive to image quality
- Heuristic-based (not as accurate as deep learning)

**Future Improvement:**

- Use pre-trained deep learning models
- Train on spoofing dataset

### 3.5 Emotion Detection Performance

**Testing:**

- Works well with clear facial expressions
- 7 emotions detected correctly in most cases
- Confidence scores provided

**Limitations:**

- Requires clear face visibility
- May struggle with ambiguous expressions
- Limited to 7 basic emotions

### 3.6 System Integration

**End-to-End Pipeline:**

1. ✅ Face detection works correctly
2. ✅ Anti-spoofing integrated seamlessly
3. ✅ Emotion detection integrated seamlessly
4. ✅ Face recognition works with database
5. ✅ UI displays all results correctly
6. ✅ Real-time attendance tracking with continuous processing

**Performance:**

- **Single Image Processing**: ~2-3 seconds per image (on CPU), ~1 second on GPU (CUDA)
- **Real-Time Tracking**:
  - Face detection: ~30 FPS (MediaPipe client-side)
  - Recognition: Every 200ms per face (throttled to avoid overload)
  - Display update: Real-time (60 FPS via requestAnimationFrame)
  - Multi-face support: Parallel processing of all detected faces

**Real-Time Tracking Architecture:**

The real-time attendance system implements a hybrid approach combining client-side and server-side processing:

1. **Client-Side (Browser)**:

   - MediaPipe Face Detection for instant face detection (no network latency)
   - Canvas overlay for real-time bounding box drawing
   - Frame capture and cropping for face regions
   - Face tracking using centroid-based algorithm

2. **Server-Side (Backend)**:

   - Face recognition using trained models
   - Anti-spoofing and emotion detection
   - Attendance logging with cooldown management
   - Screenshot saving for recognized faces

3. **Optimization Techniques**:
   - Frame resizing to 1/4 size for faster processing (similar to notebook implementation)
   - Throttled recognition requests (200ms interval) to balance accuracy and performance
   - Face ID tracking to avoid duplicate processing
   - Parallel processing of multiple faces

**Key Features:**

- **Immediate Visual Feedback**: Red boxes appear instantly when faces are detected (before recognition)
- **Color-Coded Status**:
  - Red: Unknown face or detecting
  - Green: Recognized and logged
  - Yellow: Recognized but in cooldown
  - Blue: Recognized but not yet logged
- **Automatic Logging**: Attendance automatically recorded when face is recognized (with 5-minute cooldown)
- **Screenshot Capture**: Face crops saved to `recognized_faces/` directory for audit trail
- **Live Statistics**: Real-time counters for processed, successful, and error counts

### 3.7 Challenges and Solutions

**Challenge 1: Slow Training on CPU**

- **Problem**: Full training takes 63 hours/epoch (26 days for 10 epochs)
- **Solution**:
  - Migrated to cloud GPU (Vast.AI RTX 5090)
  - Full training with all layers unfrozen
  - Reduced to 0.4-0.8 hours/epoch (8-16 hours total for 20 epochs)

**Challenge 2: Memory Issues**

- **Problem**: Large dataset (380K images) causes memory issues
- **Solution**:
  - Used DataLoader with batching
  - Implemented subset creation utility
  - Reduced batch size if needed

**Challenge 3: Training Disconnection on Cloud**

- **Problem**: Training stops when SSH connection is lost
- **Solution**: Use `screen` or `tmux` to keep training running even if disconnected

**Challenge 4: Mac Thermal Issues and System Unavailability**

- **Problem**:
  - MacBook runs extremely hot during training (fans at maximum speed)
  - System becomes unusable for other tasks (high CPU/GPU usage)
  - Risk of thermal throttling and system crashes
  - Training takes 30-50 hours, tying up MacBook for 1-2 days
  - MPS (Metal Performance Shaders) is slower than CUDA (3-5 hours/epoch vs 0.5-1 hour/epoch)
- **Solution**:
  - Migrated training to Vast.AI cloud GPU platform
  - Selected RTX 5090 ($0.37/hr) for best performance with latest generation technology
  - Reduced training time from 30-50 hours to 4-7 hours (7-12x faster)
  - Total cost: $1.50-2.60 (acceptable for saving 1-2 days)
  - MacBook remains available for other work
  - No thermal issues, dedicated GPU resources, better reliability
  - 32GB VRAM allows for larger batch sizes and future scalability
- **Result**:
  - Training completed in 4-7 hours instead of 30-50 hours
  - MacBook free for other tasks
  - No thermal throttling or system unavailability issues
  - Better performance with CUDA vs MPS
  - Latest GPU architecture for optimal training speed

**Challenge 5: Metric Learning Model Collapse**

- **Problem**:
  - Metric learning model suffered from embedding collapse
  - All embeddings became identical (cosine similarity = 1.0 for all pairs)
  - AUC dropped to 0.5397 (barely better than random)
  - Model cannot distinguish between any faces
  - Root causes: margin too small (0.5), learning rate too high (1e-4), normalization issues
- **Solution**:
  - Need to retrain with adjusted hyperparameters:
    - Increase margin to 1.0-1.5 (from 0.5)
    - Lower learning rate to 1e-5 (from 1e-4)
    - Consider removing L2 normalization during training
    - Use smaller batch size (32 instead of 64) for better triplet mining
    - Train for more epochs (30-40 instead of 20)
- **Result**:
  - Initial training failed due to collapse
  - Retraining with adjusted hyperparameters is required
  - Classification model succeeded and can be used for report

**Challenge 6: Triplet Mining Complexity**

- **Problem**: Random triplets are too easy, model doesn't learn
- **Solution**: Implemented hard negative mining (finds hardest negative for each anchor-positive pair)

**Challenge 6: Model Loading in UI**

- **Problem**: UI needs to load models, but models are large
- **Solution**: Lazy loading (load only when needed), use session state

**Challenge 7: Real-Time Face Detection and Recognition**

- **Problem**:
  - Need to detect and recognize faces in real-time from webcam feed
  - Multiple faces need to be processed simultaneously
  - MediaPipe library errors when accessing undefined properties
  - Balancing detection speed with recognition accuracy
  - Maintaining consistent face IDs across frames for tracking
- **Solution**:
  - **Hybrid Approach**: Client-side detection (MediaPipe) + server-side recognition (trained models)
  - **Error Handling**: Safe property access with null checks and default values
  - **Optimization**: Frame resizing (1/4 size) for faster processing, similar to notebook implementation
  - **Face Tracking**: Centroid-based tracking algorithm to maintain consistent IDs
  - **Throttling**: Recognition requests throttled to 200ms to balance performance and accuracy
  - **Parallel Processing**: Multiple faces processed simultaneously using async/await
  - **Immediate Feedback**: Red boxes drawn instantly when faces detected, before recognition completes
- **Implementation Details**:
  - MediaPipe Face Detection initialized with lower confidence threshold (0.3) for better detection
  - Safe extraction of bounding boxes and confidence scores with fallback defaults
  - Canvas overlay for real-time visualization without blocking video feed
  - Face ID assignment using distance-based tracking (50px threshold)
  - Cooldown period (500ms) to avoid excessive recognition requests for same face
- **Result**:
  - Real-time face detection at ~30 FPS (MediaPipe client-side)
  - Recognition processing every 200ms per face
  - Smooth visual feedback with immediate box drawing
  - Multiple faces tracked and recognized simultaneously
  - Automatic attendance logging with screenshot capture
  - System works reliably with proper error handling

### 3.8 Best Performing Approach

**Selection Criteria:**

1. AUC score (higher is better)
2. Generalization to unseen identities
3. Training stability

**Result**: **Classification Approach** is clearly the best performing method.

**Justification**:

The classification approach significantly outperforms metric learning:

1. **Superior Performance**:

   - Classification AUC: **0.9249** (excellent, exceeds 0.85 target)
   - Metric Learning AUC: **0.8747** (good, meets 0.85 target after retraining)
   - Difference: 5.02% AUC (classification is better but both are acceptable)

2. **Training Stability**:

   - Classification: Stable training, converged successfully on first attempt
   - Metric Learning: Required retraining with adjusted hyperparameters to fix initial collapse

3. **Reliability**:

   - Classification: Production-ready, consistent results, less sensitive to hyperparameters
   - Metric Learning: Good results after retraining, but more sensitive to hyperparameter choices

4. **Ease of Training**:

   - Classification: Standard supervised learning, straightforward, robust to hyperparameters
   - Metric Learning: Requires careful hyperparameter tuning (margin, learning rate, batch size)

5. **Discriminative Power**:
   - Classification: Clear separation between same/different faces (0.133 gap for cosine)
   - Metric Learning: Good separation after retraining (embeddings are discriminative)

**Conclusion**: Classification approach performs better (0.9249 vs 0.8747 AUC) and is more reliable/easier to train. However, metric learning also achieves good results (0.8747 AUC) after proper hyperparameter tuning, meeting assignment requirements. Both approaches are usable, with classification being the preferred method for production use.

---

## 4. Conclusion

### 4.1 Summary

This project successfully implements a complete face recognition attendance system with:

1. ✅ **Two face verification approaches** (Classification and Metric Learning)
2. ✅ **Anti-spoofing module** (Liveness detection)
3. ✅ **Emotion detection module** (7 emotions)
4. ✅ **User-friendly interface** (FastAPI web application)
5. ✅ **Comprehensive evaluation** (ROC curves, AUC scores)

### 4.2 Key Achievements

- ✅ Implemented both required approaches and compared their performance
- ✅ Classification model achieved excellent performance (AUC = 0.9249, exceeds 0.85 target)
- ✅ Integrated all modules into a complete system (face recognition, anti-spoofing, emotion detection)
- ✅ Created user-friendly interface for easy interaction
- ✅ Optimized training with GPU acceleration (RTX 5090 on Vast.AI)
- ✅ Comprehensive evaluation with ROC curves and AUC scores
- ✅ Honest analysis of both successes and failures
- ✅ **Real-Time Attendance Tracking**: Implemented continuous face detection and recognition system with:
  - Client-side face detection using MediaPipe (30 FPS)
  - Server-side recognition using trained models
  - Real-time bounding box visualization
  - Automatic attendance logging with cooldown management
  - Multi-face support with parallel processing
  - Screenshot capture for audit trail
  - Live statistics and attendance logs

### 4.3 Limitations

1. **Metric Learning Model**: Initially failed due to embedding collapse, but successfully recovered after retraining with adjusted hyperparameters (margin=1.0, lr=1e-5, batch_size=32, epochs=30)
2. **Anti-Spoofing**: Heuristic-based (less accurate than deep learning)
3. **Emotion Detection**: Limited to 7 basic emotions
4. **Training Time**: Requires 4-7 hours on cloud GPU (though much better than 30-50 hours on local Mac)
5. **Database**: Simple pickle storage (not scalable for large deployments)
6. **Threshold Selection**: Classification model requires careful threshold tuning (0.85) to balance false positives and false negatives when registering new faces not in training set

### 4.4 Future Improvements

1. **Anti-Spoofing**: Use pre-trained deep learning models or train custom model
2. **Emotion Detection**: Train custom model on larger dataset for more emotions
3. **Database**: Use SQLite or proper database for scalability
4. **Performance**: Optimize inference speed (model quantization, ONNX)
5. **Accuracy**: Train for more epochs, use larger models (ResNet101, EfficientNet)
6. **Threshold Optimization**: Implement adaptive threshold selection based on embedding quality
7. **Face Quality Assessment**: Add face quality scoring to reject low-quality images during registration
8. **Real-Time Tracking Enhancements**:
   - Implement face quality filtering in real-time mode
   - Add face angle validation for better accuracy
   - Optimize recognition model for faster inference (quantization, ONNX)
   - Add support for multiple cameras
   - Implement face re-identification for better tracking across frames

### 4.5 Lessons Learned

1. **Classification is More Reliable**:

   - Supervised classification with clear labels is easier to train and more stable
   - Achieved excellent results (0.9249 AUC) with standard training procedures
   - Less sensitive to hyperparameters than metric learning

2. **Metric Learning Requires Careful Tuning**:

   - Triplet loss is sensitive to margin, learning rate, and normalization
   - Model collapse is a real risk if hyperparameters are not carefully chosen (experienced this)
   - Initial training failed (AUC = 0.5397) due to margin=0.5, lr=1e-4, batch_size=64
   - Retraining with margin=1.0, lr=1e-5, batch_size=32 fixed the issue (AUC = 0.8747)
   - Hard negative mining is essential but not sufficient without proper hyperparameters
   - Requires more experimentation and tuning than classification, but can achieve good results

3. **Cloud GPU is Essential for Large-Scale Training**:

   - Local Mac M2 training was impractical (30-50 hours, thermal issues)
   - Cloud GPU (RTX 5090) provided 7-12x speedup at reasonable cost ($3-6)

4. **Real-Time Processing Requires Hybrid Architecture**:

   - **Client-Side Detection**: MediaPipe in browser provides instant face detection (30 FPS) without network latency
   - **Server-Side Recognition**: Trained models on backend ensure accuracy and security
   - **Optimization is Critical**: Frame resizing, throttling, and parallel processing are essential for smooth performance
   - **Error Handling**: Safe property access and null checks prevent crashes from library inconsistencies
   - **Face Tracking**: Centroid-based tracking maintains consistent IDs across frames for better user experience
   - **Immediate Feedback**: Drawing boxes instantly (before recognition) provides better UX than waiting for server response
   - **Multi-Face Support**: Parallel processing allows handling multiple faces simultaneously without blocking
   - **Best Practice**: Learned from existing implementations (notebook) and adapted to web environment

5. **Honest Reporting is Important**:

   - Acknowledging failures (metric learning collapse) shows understanding
   - Discussing root causes and solutions demonstrates learning
   - Not all experiments succeed, and that's valuable information

6. **Hyperparameter Sensitivity**:

   - Classification: Robust to hyperparameter choices
   - Metric Learning: Very sensitive, requires careful tuning
   - Margin, learning rate, and normalization all critical for metric learning

7. **Evaluation Metrics Matter**:

   - AUC is better than accuracy for verification tasks
   - ROC curves reveal model quality across all thresholds
   - Mean similarity values help diagnose model issues (collapse detection)

8. **Threshold Selection is Critical**:
   - Evaluation threshold (0.7) may differ from production threshold (0.85)
   - Higher thresholds reduce false positives but may increase false negatives
   - Production systems require careful threshold tuning based on use case
   - Classification model works well with 0.85 threshold for new face registration

### 4.6 Lessons Learned (Original)

1. **Transfer Learning**: Pre-trained models (ResNet50) provide excellent starting point
2. **GPU Acceleration**: Essential for deep learning projects. Cloud GPUs (RTX 5090) provide 7-12x speedup compared to local training, with better reliability and no thermal issues.
3. **Cloud GPU for Training**: For long training sessions, cloud GPUs (Vast.AI) offer better performance, reliability, and system availability than local machines. RTX 5090 at $1.50-2.60 is cost-effective for saving 1-2 days of training time, with latest generation technology providing 7-12x speedup.
4. **Thermal Management**: Local training on laptops causes severe thermal issues (max fans, system unavailability). Cloud GPUs eliminate these concerns.
5. **Modular Design**: Separating components makes testing and debugging easier
6. **Evaluation**: ROC/AUC are better metrics than accuracy for verification tasks
7. **Hard Negative Mining**: Critical for metric learning (random triplets don't work)

---

## 5. References

1. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. _arXiv preprint arXiv:1503.03832_.

2. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_.

4. Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. _IEEE Signal Processing Letters_.

5. Goodfellow, I., et al. (2013). Challenges in representation learning: A report on the machine learning competition. _International Conference on Machine Learning_.

---

## Appendix A: Project Structure

```
facial-recognition/
├── src/
│   ├── models/
│   │   ├── classification_model.py
│   │   └── metric_learning_model.py
│   ├── modules/
│   │   ├── anti_spoofing.py
│   │   └── emotion_detection.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── evaluation.py
│   │   ├── face_detector.py
│   │   ├── face_database.py
│   │   └── triplet_mining.py
│   ├── train_classification.py
│   ├── train_metric_learning.py
│   ├── evaluate_classification.py
│   ├── evaluate_metric_learning.py
│   └── app.py
├── models/              # Trained model checkpoints
├── results/             # Evaluation results and plots
└── requirements.txt     # Dependencies
```

---

## Appendix B: Hyperparameters

**Classification Model:**

- Learning Rate: 1e-4
- Batch Size: 32
- Epochs: 10
- Optimizer: Adam
- Loss: Cross-entropy
- Learning Rate Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)

**Metric Learning Model:**

- Learning Rate: 1e-4
- Batch Size: 32
- Epochs: 10
- Optimizer: Adam
- Loss: Triplet Loss (margin=0.5)
- Embedding Dimension: 512
- Learning Rate Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)

---

## Appendix C: Training Commands

**Train Classification Model:**

```bash
python src/train_classification.py \
    --data_dir classification_data \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_dir models
```

**Train Metric Learning Model:**

```bash
python src/train_metric_learning.py \
    --data_dir classification_data \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --margin 0.5 \
    --embedding_dim 512 \
    --save_dir models
```

---

**End of Report**
