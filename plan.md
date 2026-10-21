# Project Execution Plan

## Overview

Execution strategy aligning technical architecture with implementation milestones, transitioning from initial data pipeline preparation to repository setup, training verification, interpretability, and production-ready packaging.

## Phase 1: Foundation
Goal: Establish project architecture, environment setup, data pipeline, and version control repository.
Steps:
1. Initialize project folder structure and python virtual environment.
2. Implement split-folders pipeline (`data_split.py`) for train/val/test segregation.
3. Configure `.gitignore` to prevent tracking of multi-gigabyte raw/processed datasets and cache files.
4. Establish GitHub remote tracking and perform initial commit and push.
Status: In Progress

## Phase 2: Core Development
Goal: Build robust neural network training and evaluation pipelines.
Steps:
1. Implement `src/config.py` with centralized hyperparameters and directory paths.
2. Develop `src/data_prep.py` with `tf.data` prefetch, augmentations, and batching.
3. Construct two-stage transfer learning training script `src/train.py` using EfficientNetV2L.
4. Build `src/evaluate.py` to calculate multi-class confusion matrix and classification metrics.
5. Create `src/infer.py` for inference on individual scan files.
Status: Complete (Code implemented; weights pending full training)

## Phase 3: Interpretability & Analysis
Goal: Provide explainability for clinical diagnostics and exploratory data analysis.
Steps:
1. Complete Exploratory Data Analysis in `notebooks/EDA.ipynb`.
2. Implement Grad-CAM in `src/gradcam.py` to highlight focal brain regions contributing to dementia classifications.
3. Add utility functions in `src/utils.py` for image loading and visualization overlays.
Status: Pending

## Phase 4: Testing & Model Validation
Goal: Validate model accuracy, generalization across test splits, and latency benchmarks.
Steps:
1. Run complete model training run and log metrics to TensorBoard.
2. Evaluate performance across all 4 classes on the unseen test partition.
3. Benchmark inference execution time on CPU and GPU.
Status: Pending

## Phase 5: Deployment & Packaging
Goal: Package the system for standalone inference or web API demonstration.
Steps:
1. Provide a lightweight inference API or demo interface.
2. Document environment setup and reproduction steps.
3. Publish final project release and tag.
Status: Pending

## Completed Milestones

- Project directory and modular source code structure initialized
- Data preprocessing and two-stage training pipeline designed
- Git repository connected to GitHub origin remote

## Pending Milestones

- Push code, configuration, and documentation to GitHub repository
- Implement Grad-CAM model explainability module (`src/gradcam.py`)
- Complete exploratory analysis in `notebooks/EDA.ipynb`
- Train model weights and evaluate full test metrics
