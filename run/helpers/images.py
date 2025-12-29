
import glob
import os
import base64
from pathlib import Path

def get_latest_image(folder_name):
    list_of_files = glob.glob(folder_name) 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def image_file_to_base64(path: str):
    """
    Read an image from disk and return (media_type, base64_str)
    suitable for Claude's image source block.
    """
    img_path = Path(path)
    ext = img_path.suffix.lower()

    if ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"
    elif ext == ".gif":
        media_type = "image/gif"
    else:
        # default/fallback
        media_type = "image/png"

    with img_path.open("rb") as f:
        img_bytes = f.read()

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return media_type, b64


def image_file_to_data_url(path: str) -> str:
    """
    Read an image from disk and return a data URL string suitable
    for Groq / OpenAI-style image_url usage.
    """
    img_path = Path(path)
    # crude MIME guess from suffix; tweak if you know it's always jpg/png
    ext = img_path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        # fall back to jpeg if unknown
        mime = "image/jpeg"

    with img_path.open("rb") as f:
        img_bytes = f.read()

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"