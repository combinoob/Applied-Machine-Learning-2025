import os

SEED = 42  # Change this to update train/val/test split
RAW_DATA_PATH = "data/raw_data.csv"
TRAIN_DATA_PATH = "data/train.csv"
VAL_DATA_PATH = "data/validation.csv"
TEST_DATA_PATH = "data/test.csv"
DVC_REMOTE = "gdrive://your-google-drive-id"
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "benchmark_models"