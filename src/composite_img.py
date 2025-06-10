from PIL import Image


def make_composite_grid(img_paths, target_h=512, target_w=512):
    """
    Create a 2x2 grid composite image from a list of image paths.
    Each image is padded or resized to the target height and width.
    """
    imgs = []
    for p in img_paths:
        im = Image.open(p).convert("RGB")
        print(f"Processing image before resizing: {p} with size {im.size}")
        if im.width > target_w or im.height > target_h:
            # Resize if the image is larger than the target dimensions
            im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            # Pad the image to the target dimensions
            padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))  # Black padding
            x_offset = (target_w - im.width) // 2
            y_offset = (target_h - im.height) // 2
            padded.paste(im, (x_offset, y_offset))
            im = padded

        print(f"Processed image size: {im.size}")
        imgs.append(im)

    # Calculate dimensions for the 2x2 grid
    grid_width = target_w * 2
    grid_height = target_h * 2

    grid = Image.new("RGB", (grid_width, grid_height))

    # Arrange images in the grid
    x_positions = [0, target_w]
    y_positions = [0, target_h]

    for i, im in enumerate(imgs):
        x = x_positions[i % 2]
        y = y_positions[i // 2]
        grid.paste(im, (x, y))

    return grid
