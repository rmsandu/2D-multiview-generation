import sys

from dotenv import load_dotenv
from pathlib import Path
import random, base64, json, io, os
from PIL import Image
from google import genai
from google.genai import types
from tqdm import tqdm


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-2.0-flash"
CACHE_DIR = Path(".gemini_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _caption_one(img_pil, category):
    prompt = (
        f"You are describing only a single "
        f"{category} object for multi-view dataset curation. Focus only on the object described by the category name {category}. "
        f"Respond with a comprehensive detailed description of the object using ONLY the name from {category}, including the viewing angle of the camera, ideally in degrees."
        "EXAMPLE ANSWER if category is backpack: This photo shows a 0-degree angle front-view shot of a blue of backpack with a front pocket. "
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[img_pil, prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=100,  # limit to 50 tokens
            temperature=0.4,  # low temperature for consistency
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
            _caption_one(Image.open(p).convert("RGB"), category) for p in view_paths
        ]
        cache_file.write_text(json.dumps(clauses), encoding="utf-8")

    joint = (
        "[FOUR-VIEWS] This set of four images shows different viewing angles of the same "
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
    print("Captioner module loaded. Use caption_four_views() to caption images.")
    if len(sys.argv) > 1:
        print("Usage: python captioner.py")
        print("This module is not meant to be run directly. Use it as a library.")
        sys.exit(1)
