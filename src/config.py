import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# Model Paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_final_model.pkl")

# Random State
RANDOM_STATE = 42
