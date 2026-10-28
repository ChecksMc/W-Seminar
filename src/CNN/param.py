"""Typed default configuration and project data paths for the CNN workflow."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw" / "emoart5k"
IMAGE_DIR: Path = RAW_DIR / "Images"
ANNOTATION_FILE: Path = RAW_DIR / "annotation.json"
SPLIT_FILE: Path = PROJECT_ROOT / "data" / "splits" / "emoart5k_70_15_15.csv"
OUTPUT_DIR: Path = PROJECT_ROOT / "runs" / "cnn"
VISUALIZATION_DIR: Path = PROJECT_ROOT / "runs" / "visualization"

IMAGE_SIZE: int = 128
BATCH_SIZE: int = 32
EPOCHS: int = 15
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
EMBEDDING_DIM: int = 64
DROPOUT: float = 0.25
RANDOM_SEED: int = 42
TRAIN_FRACTION: float = 0.70
VALIDATION_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15
K_NEIGHBORS: int = 12
EMBEDDING_WEIGHT: float = 0.5
RECOGNITION_WEIGHT: float = 0.5
