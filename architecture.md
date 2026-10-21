# Project Architecture

## Overview

Deep learning system designed for the automated detection and 4-stage classification of Alzheimer's Disease from structural brain MRI images using transfer learning with EfficientNetV2L.

## Tech Stack

Frontend:
N/A (CLI and programmatic inference pipelines)

Backend:
Python 3.10+, TensorFlow 2.x, Keras

Database:
Local file system storage for raw/processed MRI datasets

Authentication:
N/A

Hosting:
GitHub (Version control & repository hosting)

Other Tools:
Split-Folders, OpenCV, NumPy, Scikit-learn, TensorBoard

## System Components

### Frontend
Not applicable. The interface is currently exposed via command-line inference scripts (`src/infer.py`) and evaluation pipelines (`src/evaluate.py`).

### Backend
Comprises the core model architecture and training engine:
- Model Backbone: Pre-trained EfficientNetV2L (ImageNet weights) with native input preprocessing.
- Classification Head: Global Average Pooling, Dropout (0.3), Softmax Dense Layer (4 classes).
- Two-Stage Training Engine:
  - Stage 1: Feature extractor frozen, training top dense classification head (Adam optimizer, lr=1e-4, Label Smoothing=0.05).
  - Stage 2: Fine-tuning the top 60 layers of EfficientNetV2L with reduced learning rate (lr=2e-5).
- Data Augmentation & Ingestion: `tf.data` pipeline with horizontal flips, small rotations, zooming, and contrast adjustments.

### Database
File-system-based image dataset hierarchy:
- Raw dataset: Structural brain MRI scans divided into 4 diagnostic categories (`NonDemented`, `VeryMildDementia`, `MildDementia`, `ModerateDementia`).
- Processed dataset: Ratio split (64% train, 16% validation, 20% test).

### APIs
Internal modular Python interfaces:
- `src.data_prep.make_datasets()`: Generates batched, augmented `tf.data.Dataset` streams.
- `src.infer.predict(img_path)`: Loads model and returns predicted diagnostic label, confidence score, and class probabilities.
- `src.evaluate.main()`: Computes multi-class classification metrics and confusion matrix.

## Folder Structure

```
alzheimers_detection/
├── .gitignore             # Excludes large binaries, datasets, environments
├── architecture.md        # Technical architecture and system design
├── current.md             # Real-time project status and changes
├── data/                  # MRI dataset (raw and processed) - gitignored
│   ├── processed/         # Train, val, test splits
│   └── raw/               # Original downloaded scans
├── data_split.py          # Dataset splitting script (splitfolders)
├── logs/                  # Training logs and TensorBoard telemetry
├── models/                # Saved model weights (.keras / .h5)
│   ├── checkpoints/       # Best epoch checkpoints
│   └── effv2l_alzheimers.h5
├── notebooks/             # Exploratory Data Analysis & experiments
├── plan.md                # Project execution plan and milestones
├── requirements.txt       # Project dependencies
├── rules.md               # Documentation governance rules
└── src/                   # Core source package
    ├── config.py          # Centralized hyperparameters and path configs
    ├── data_prep.py       # Data pipeline and augmentation
    ├── evaluate.py        # Model evaluation and confusion matrix
    ├── gradcam.py         # Visual interpretability via Grad-CAM
    ├── infer.py           # Single-image inference module
    ├── train.py           # Two-stage training and fine-tuning loop
    └── utils.py           # Helper utilities
```

## Data Flow

1. Raw MRI scan images are ingested from `data/raw/dataset/Data`.
2. `data_split.py` splits the dataset deterministically (seed=42) into train (64%), val (16%), and test (20%) sets under `data/processed/`.
3. `src/data_prep.py` constructs `tf.data` pipelines with RGB conversion, target size resizing (480x480), data augmentations, batching, and AUTOTUNE prefetching.
4. `src/train.py` feeds batches through EfficientNetV2L in two stages (frozen backbone -> fine-tuned top 60 layers) while logging callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard).
5. The trained model is saved under `models/effv2l_alzheimers.h5`.
6. `src/evaluate.py` runs validation on the test set, computing class precision, recall, F1-score, and confusion matrices.
7. `src/infer.py` accepts single MRI scans and outputs classification predictions with confidence levels.

## Architecture Decisions

Decision 1: Use EfficientNetV2L as transfer learning backbone
Reason: EfficientNetV2 offers high parameter efficiency, fast training convergence, and superior feature representation for subtle textural variations in medical neuroimaging compared to earlier ResNet or VGG architectures.

Decision 2: Exclude raw datasets and heavy model weights from Git version control
Reason: The MRI dataset comprises 170k+ images (>3.8 GB) which exceeds GitHub's 2GB push limit and violates standard Git repository best practices. Datasets are managed locally and reproduced via deterministic preprocessing pipelines.

Decision 3: Two-stage training with label smoothing
Reason: Transfer learning on medical images risks destroying pre-trained weights if trained end-to-end with high initial learning rates. Freezing the base first stabilizes the classifier head. Label smoothing (0.05) prevents overconfidence across borderline dementia stages.
