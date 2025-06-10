# --- config.py ---
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import traceback
import pprint
import sys
from .captioner import (
    caption_four_views,
    generate_caption_composite_grid,
)
from .composite_img import make_composite_grid

# --- directory layout ---
DATA_ROOT = Path("data")
OBJECTS_DIR = DATA_ROOT / "mvi_32"  # change to mvi_40 if needed
OUT_DIR = Path("training/composites_4view")
OUT_DIR_GRID = Path("training/composites_4view_grid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_GRID.mkdir(parents=True, exist_ok=True)

CATEGORY_FILE = Path(DATA_ROOT / "mvimgnet_category.txt")

# --- processing params ---
N_OBJECTS = 63  # MVP size
TARGET_HEIGHT = 512  # each tile resized to this height
MODEL_NAME = "gemini-2.0-flash-preview"
ANGLE_TAGS = ["front", "side", "back", "top"]
TARGET_HEIGHT = 512  # each tile resized to this height

# Gemini key is read from env var GOOGLE_API_KEY


def find_image_dirs(root: Path):
    """Return sorted list of leaf 'images' directories."""
    return sorted(p for p in root.rglob("images") if p.is_dir())


def load_categories(txt: Path):
    """Load category map from MVimgnet text file of categories."""
    if not txt.exists():
        print(f"Category file {txt} does not exist. Returning empty map.")
        return {}
    return dict(
        line.strip().split(",", maxsplit=1)
        for line in txt.read_text(encoding="utf-8").splitlines()
        if "," in line  # Ensure the line contains a comma
    )


def choose_four_views(img_paths):
    """Pick indices 0, 1/4, 1/2, last one through the sweep."""
    paths = sorted(img_paths)
    n = len(paths)
    idx = [0, n // 4, n // 2, n - 2]
    idx = [min(i, n - 1) for i in idx]
    picks = [paths[i] for i in idx]
    while len(picks) < 4:  # handle <4 frames edge-case
        picks.append(picks[-1])
    return picks


def process_one(img_dir, id2cat, cache_dir):
    """
    Process one object directory to create composite image and caption.
    """

    # pprint.pprint(id2cat)
    obj_id = str(img_dir.parent.parent.name)
    category = id2cat.get(obj_id, "object")

    if category == "object":
        print(
            f"Warning: Category for {obj_id} not found in {img_dir.parent.name}, using 'object'."
        )

    # replace trailing spaces from category name with "-"
    category = category.strip().replace(" ", "-")
    # Select four views
    views = choose_four_views(img_dir.glob("*.jpg"))

    # Generate captions with Google Gemini Flash 2.0 from captioner.py

    # _, single_caption = caption_four_views(
    #     view_paths=views,  # Corrected argument name
    #     category_name=category,
    #     obj_id=obj_id,
    #     folder_id=img_dir.parent.name,
    # )

    # Create composite image and composite prompt
    composite = make_composite_grid(views, target_h=TARGET_HEIGHT)

    # Generate caption for the composite image 2x2 grid
    joint_caption = generate_caption_composite_grid(composite, category)

    # Save composite image and caption with the folder name as prefix
    composite_file = cache_dir / f"{category}_{obj_id}_{img_dir.parent.name}.png"
    caption_file = cache_dir / f"{category}_{obj_id}_{img_dir.parent.name}.txt"

    composite.save(composite_file)
    caption_file.write_text(joint_caption, encoding="utf-8")

    print(f"Saved composite image to {composite_file}")
    print(f"Saved caption to {caption_file}")


def main():
    id2cat = load_categories(CATEGORY_FILE)
    all_dirs = find_image_dirs(OBJECTS_DIR)[:N_OBJECTS]

    for img_dir in tqdm(all_dirs, desc="Processing objects"):
        try:
            process_one(img_dir, id2cat, cache_dir=OUT_DIR_GRID)
        except Exception as e:
            print(f"Failed on {img_dir}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
