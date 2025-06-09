# --- config.py ---
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --- directory layout ---
DATA_ROOT = Path("data")
OBJECTS_DIR = DATA_ROOT / "mvi_32"  # change to mvi_40 if needed
OUT_DIR = Path("training/composites_4view")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_FILE = "mvimgnet_categories.txt"

# --- processing params ---
N_OBJECTS = 60  # MVP size
TARGET_HEIGHT = 512  # each tile resized to this height
MODEL_NAME = "gemini-2.5-flash-preview"
ANGLE_TAGS = ["front", "side", "back", "top"]
TARGET_HEIGHT = 512  # each tile resized to this height

# Gemini key is read from env var GOOGLE_API_KEY


def find_image_dirs(root: Path):
    """Return sorted list of leaf 'images' directories."""
    return sorted(p for p in root.rglob("images") if p.is_dir())


def load_categories(txt: Path):
    """Load category map from MVimgnet text file of categories."""
    if not txt.exists():
        return {}
    return dict(line.strip().split(maxsplit=1) for line in txt.read_text().splitlines())


def choose_four(img_paths):
    """Pick indices 0, 1/4, 1/2, 3/4 through the sweep."""
    paths = sorted(img_paths)
    n = len(paths)
    idx = [0, n // 4, n // 2, 3 * n // 4]
    idx = [min(i, n - 1) for i in idx]
    picks = [paths[i] for i in idx]
    while len(picks) < 4:  # handle <4 frames edge-case
        picks.append(picks[-1])
    return picks


def test_choose_four(tmp_path):
    # fabricate 10 dummy filenames
    paths = [tmp_path / f"{i:03d}.jpg" for i in range(10)]
    for p in paths:
        p.touch()
    sel = choose_four(paths)
    assert len(sel) == 4 and sel[0].name == "000.jpg"


if __name__ == "__main__":
    id2cat = load_category_map(CATEGORY_FILE)
    img_dirs = find_image_dirs(OBJECTS_DIR)[:N_OBJECTS]
    for img_dir in tqdm(img_dirs, desc="objects"):
        try:
            obj_id = img_dir.parent.name
            category = id2cat.get(obj_id, "object")

            views = choose_four(img_dir.glob("*.jpg"))
            clauses, joint_caption = caption_four_views(views, category)

            composite = make_composite(views, target_h=TARGET_H)
            composite.save(OUT_DIR / f"{obj_id}.png", "PNG")
            (OUT_DIR / f"{obj_id}.txt").write_text(joint_caption, encoding="utf-8")
        except Exception:
            print(f"⚠️  Failed on {img_dir}")
            traceback.print_exc()
