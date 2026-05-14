from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

NIFTY_FILE = DATA_DIR / "nifty50.csv"
FEATURE_FILE = DATA_DIR / "starter_features.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42
