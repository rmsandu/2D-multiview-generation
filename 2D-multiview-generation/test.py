from PIL import Image, ImageDraw
import json, os, random

os.makedirs("data/debug/images", exist_ok=True)
with open("data/debug/train.jsonl", "w") as f:
    for i in range(50):  # 50 synthetic samples
        img = Image.new("RGB", (512 * 2, 512), "white")
        d = ImageDraw.Draw(img)
        d.text((20, 200), f"OBJ{i}-A", fill="black")
        d.text((532, 200), f"OBJ{i}-B", fill="black")
        name = f"d{i}.png"
        img.save(f"data/debug/images/{name}")
        prompt = (
            "[TWO-VIEWS] Two orthogonal views of a test object. "
            "[IMAGE1] Front view. [IMAGE2] Side view."
        )
        f.write(json.dumps({"file": name, "prompt": prompt}) + "\n")


pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
