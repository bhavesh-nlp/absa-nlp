from pathlib import Path

# Base project directory (root of your repository)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Model directories
MODEL_DIR = BASE_DIR / "models"
ASPECT_MODEL_DIR = MODEL_DIR / "aspect_extraction"
SENTIMENT_MODEL_DIR = MODEL_DIR / "sentiment_classification"

# Ensure required folders exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ASPECT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
SENTIMENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
