# rename all files pairs img - txt from 001 and so on...
from pathlib import Path


def rename_files_in_directory(directory):
    """
    Rename all image and text file pairs in the specified directory to sequential numbers starting from 0001.
    """
    files = sorted(Path(directory).glob("*"))
    img_files = [f for f in files if f.suffix in [".png", ".jpg", ".jpeg"]]
    txt_files = [f for f in files if f.suffix == ".txt"]

    if len(img_files) != len(txt_files):
        print("Warning: The number of image files and text files do not match.")

    for i, (img, txt) in enumerate(zip(img_files, txt_files), start=64):
        new_img_name = f"{i:04d}{img.suffix}"
        new_txt_name = f"{i:04d}.txt"

        img.rename(img.parent / new_img_name)
        txt.rename(txt.parent / new_txt_name)

        print(f"Renamed {img.name} to {new_img_name}")
        print(f"Renamed {txt.name} to {new_txt_name}")


if __name__ == "__main__":
    target_directory = Path("training/composites_4view_grid")
    rename_files_in_directory(target_directory)
