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
        f"You are describing a photograph of a single  "
        f"{category} object for multi-view dataset curation. Focus only on the object described by the category name {category}. "
        f"Respond with a comprehensive detailed description of the object using ONLY the name from {category}, including the viewing angle of the camera, ideally in degrees."
        "EXAMPLE ANSWER if category is backpack: This photo shows a 0-degree angle front-view shot of a blue of backpack with a front pocket and two zippers. "
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[img_pil, prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=100,  # limit to 50 tokens
            temperature=0.4,  # low temperature for consistency
        ),
    )
    # print(resp.text)
    return resp.text.strip().rstrip(".")  # remove trailing period if any


def caption_four_views(
    view_paths, category_name, obj_id, folder_id
) -> tuple[list[str], str]:
    """
    Returns (clauses list[4], IC-LoRA joint caption)
    caches individual clauses in .gemini_cache/<hash>.json
    """

    unique_prefix = f"{category_name}_{obj_id}_{folder_id}_"
    cache_key = f"{unique_prefix}-" + "-".join(p.stem for p in view_paths) + ".json"
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists():
        clauses = json.loads(cache_file.read_text())
    else:
        clauses = [
            _caption_one(Image.open(p).convert("RGB"), category_name)
            for p in view_paths
        ]
        cache_file.write_text(json.dumps(clauses), encoding="utf-8")

    joint = (
        (
            f"[FOUR-VIEWS] This set of four images shows different viewing angles of the same {category_name}; "
            + "; ".join(
                f"[IMAGE{i+1}] {c.replace('Here is a description of the object in the image:', '').strip()}"
                for i, c in enumerate(clauses)
            )
            + "."
        )
        .replace("\n", "")
        .replace("  ", " ")
    )
    return clauses, joint


def generate_caption_composite_grid(composite_img_pil, category):
    """
    Generate a caption for a composite image of four views.
    This function is a placeholder and should be implemented as needed.
    """
    prompt = (
        f"You are an expert data annotator for 3D computer vision."
        f'Your task is to generate a precise, single-line "joint caption" for a 2x2 grid of images showing different views of the same object.'
        f"The image I am providing is a 2x2 grid image that represents a different camera angle of the same object. "
        f"All four views show the same object, which belongs to the category: {category}."
        f"Your task is to analyze the camera angle of each of the four images in the grid and create a single, continuous line of text that describes the object and the specific viewpoint in each grid position."
        f"You must follow the strict formatting rules for the caption: [FOUR-VIEWS]  This set of four images show SHORT DESCRIPTION of object {category} in the photo; [TOP-LEFT], [TOP-RIGHT], [BOTTOM-LEFT], and [BOTTOM-RIGHT]"
        f"EXAMPLE ANSWER: [FOUR-VIEWS] This set of four images image shows different viewing angles of the same blue bag wtih a flower pattern; [TOP-LEFT] This photo shows a 45-degree angle shot of a blue bag; [TOP-RIGHT] This photo shows high-angle view shot of a blue bag; [BOTTOM-LEFT] This photo shows another side view shot of a blue bag with a dragon on it; [BOTTOM-RIGHT] This photo shows the back view of a blue bag with two straps."
    )

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[composite_img_pil, prompt],
        config=types.GenerateContentConfig(
            temperature=0.3,  # low temperature for consistency
        ),
    )
    # print(resp.text)
    return resp.text.strip().rstrip(".")  # remove trailing period if any


if __name__ == "__main__":
    print("Captioner module loaded. Use caption_four_views() to caption images.")
    if len(sys.argv) > 1:
        print("Usage: python captioner.py")
        print("This module is not meant to be run directly. Use it as a library.")
        sys.exit(1)

    img_dir = Path("/mnt/data/4-views/00000/")
    obj_id = img_dir.parent.parent.name
    print(f"Extracted obj_id: {obj_id}")
