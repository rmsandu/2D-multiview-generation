# --- config.py ---
from pathlib import Path

# --- directory layout ---
DATA_ROOT = Path("data")
OBJECTS_DIR = DATA_ROOT / "mvi_32"  # change to mvi_40 if needed
OUT_DIR = Path("training/composites_4view")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_FILE = DATA_ROOT / "mvimgnet_categories.txt"  # optional

# --- processing params ---
N_OBJECTS = 60  # MVP size
TARGET_HEIGHT = 512  # each tile resized to this height
MODEL_NAME = "gemini-2.5-flash-preview"
ANGLE_TAGS = ["front", "side", "back", "top"]

# Gemini key is read from env var GOOGLE_API_KEY
