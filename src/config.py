import os

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
CKPT_DIR  = os.path.join(MODEL_DIR, "checkpoints")
LOG_DIR   = os.path.join(PROJECT_DIR, "logs", "tensorboard")

# Data
CLASSES = ["NonDemented", "VeryMildDementia", "MildDementia", "ModerateDementia"]
IMG_SIZE = (480, 480)
BATCH_SIZE = 8   # 8 or 16 depending on GPU RAM
SEED = 42

# Training
BASE_LR = 1e-4
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
