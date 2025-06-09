# common imports
from dotenv import load_dotenv
from pathlib import Path
import random, base64, json, io, os
from PIL import Image
from google import genai
from google.genai import types
from tqdm import tqdm
from dataset_builder import OBJECTS_DIR, choose_four, find_image_dirs


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-2.0-flash"
CATEGORY_TAGS = "mvimgnet_category.txt"
ANGLE_TAGS = ["front", "side", "back", "top"]
CACHE_DIR = Path(".gemini_cache")
CACHE_DIR.mkdir(exist_ok=True)

# """ """ prompt = (
#     f"You are describing a {angle}-view photograph of a single "
#     f"{category} object for multi-view dataset curation. "
#     f"Respond with one short description that "
#     f"mentions colour/shape and the angle."
#     "Here is an example of answer: [FOUR-VIEWS] This set of four images show four different angles of the same DSLR camera; "
#     "[IMAGE1] front view (0 degrees) of the same camera, black body with silver lens mount;"
#     "[IMAGE2] side view (45 degrees) of the camera showing hand-grip and shutter button;"
#     "[IMAGE3] side view (90 degrees) of the camera revealing LCD screen and dials;"
#     "[IMAGE4] top view (above) of the camera highlighting hot-shoe and mode dial."
# )
#  """


def _caption_one(img_pil, angle, category):
    prompt = (
        f"You are describing a {angle}-view photograph of a single "
        f"{category} object for multi-view dataset curation. "
        f"Respond with one short description that mentions colour/shape and the viewing angle."
        f"Example of a good response: "
        f"the front view of a black DSLR camera with a silver lens mount"
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[img_pil, prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=50,  # limit to 50 tokens
            temperature=0.2,  # low temperature for consistency
        ),
    )
    print(resp.text)
    return resp.text.strip().rstrip(".")  # remove trailing period if any


def caption_four_views(view_paths, category) -> tuple[list[str], str]:
    """
    Returns (clauses list[4], IC-LoRA joint caption)
    caches individual clauses in .gemini_cache/<hash>.json
    """
    cache_key = "-".join(p.stem for p in view_paths) + ".json"
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists():
        clauses = json.loads(cache_file.read_text())
    else:
        clauses = [
            _caption_one(Image.open(p).convert("RGB"), ang, category)
            for p, ang in zip(view_paths, ANGLE_TAGS)
        ]
        cache_file.write_text(json.dumps(clauses), encoding="utf-8")

    joint = (
        "[FOUR-VIEWS] This  set of four image shows different viewing angles of the same "
        f"{category}; "
        + "; ".join(f"[IMAGE{i+1}] {c}" for i, c in enumerate(clauses))
        + "."
    )
    return clauses, joint


def make_composite(img_paths, target_h=512):
    """
    Create a horizontal strip composite image from a list of image paths.
    Each image is resized to the target height while maintaining aspect ratio."""
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    scale = target_h / imgs[0].height
    imgs = [im.resize((int(im.width * scale), target_h), Image.LANCZOS) for im in imgs]
    W = sum(im.width for im in imgs)
    strip = Image.new("RGB", (W, target_h))
    x = 0
    for im in imgs:
        strip.paste(im, (x, 0))
        x += im.width
    return strip


if __name__ == "__main__":

    # grab FIRST object
    first_dir = find_image_dirs(OBJECTS_DIR)[0]
    obj_id = first_dir.parent.name
    # load category from file if available
    category_file = OBJECTS_DIR / CATEGORY_TAGS
    if category_file.exists():
        id2cat = dict(
            line.strip().split(maxsplit=1)
            for line in category_file.read_text().splitlines()
        )
        category = id2cat.get(obj_id, "object")
    else:
        category = "object"

    views = choose_four(first_dir.glob("*.jpg"))

    clauses, joint_caption = caption_four_views(views, category)
    print("Generated clauses:", clauses)
    print("\nJoint caption:\n", joint_caption)

    composite = make_composite(views, target_h=512)
    out_file = Path(f"quick_test_{obj_id}.png")
    composite.save(out_file)
    print(f"\nSaved composite strip to {out_file.resolve()}")
    composite.show()  # opens preview window
