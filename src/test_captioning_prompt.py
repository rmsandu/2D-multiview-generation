from .dataset_builder import OBJECTS_DIR, choose_four_views, find_image_dirs
from .captioner import caption_four_views, make_composite
from pathlib import Path
from PIL import Image
import os

# grab FIRST object
first_dir = find_image_dirs(OBJECTS_DIR)[0]
obj_id = first_dir.parent.name
# Infer category from folder name using mvimgnet_category.txt
CATEGORY_TAGS = "mvimgnet_category.txt"
category_file = Path(CATEGORY_TAGS)

if category_file.exists():
    id2cat = dict(
        line.strip().split(",", maxsplit=1)
        for line in category_file.read_text(encoding="utf-8").splitlines()
    )
    category = id2cat.get(obj_id, "object")
else:
    category = "object"

views = choose_four_views(first_dir.glob("*.jpg"))

# Updated call to caption_four_views to include missing arguments
clauses, joint_caption = caption_four_views(
    views, category, obj_id=obj_id, folder_id=first_dir.parent.name
)
print("Generated clauses:", clauses)
print("\nJoint caption:\n", joint_caption)

composite = make_composite(views, target_h=512)

# Save composite image and caption
out_file = Path(f"training/composites_4view/{category}_{obj_id}.png")
out_file.parent.mkdir(parents=True, exist_ok=True)
composite.save(out_file)

caption_file = Path(f"training/composites_4view/{category}_{obj_id}.txt")
caption_file.write_text(joint_caption, encoding="utf-8")

print(f"\nSaved composite strip to {out_file.resolve()}")
print(f"Saved caption to {caption_file.resolve()}")
composite.show()  # opens preview window
