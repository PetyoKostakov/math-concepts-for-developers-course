import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# Output base directory
base_dir = "generated_digits"

# Folder structure
folders = {
    "centered": {"rotate": False, "vary_size": False, "vary_position": False,
                 "base_size": 24, "margin": 1},
    "different_sizes": {"rotate": False, "vary_size": True, "vary_position": False,
                        "min_size": 16, "max_size": 28, "margin": 1},
    "different_rotations": {"rotate": True, "vary_size": False, "vary_position": False,
                            "base_size": 22, "max_angle": 30, "margin": 1},
    "mixed": {"rotate": True, "vary_size": True, "vary_position": True,
              "min_size": 14, "max_size": 20, "max_angle": 20, "margin": 1},
    "mixed_test": {"rotate": True, "vary_size": True, "vary_position": True,
                   "min_size": 14, "max_size": 20, "max_angle": 20, "margin": 1},
}

IMG_SIZE = 28

FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
try:
    import matplotlib
    FONT_PATHS.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"))
except ImportError:
    pass


def load_font(font_size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_image(digit, rotate=False, vary_size=False, vary_position=False,
                   min_size=8, max_size=18, max_angle=45, margin=None, base_size=14):
    """Generate one digit image.

    If `margin` is set the image is regenerated until the digit is fully inside
    the image with at least `margin` black pixels around it (no clipping).
    """
    for _ in range(500):
        img = _draw_digit(digit, rotate, vary_size, vary_position, min_size, max_size, max_angle, base_size)
        if margin is None or _fits_inside(img, margin):
            return img
    raise RuntimeError(f"Could not generate unclipped image of {digit}")


def _fits_inside(img, margin):
    ink = np.argwhere(np.asarray(img) > 0)
    top, left = ink.min(axis=0)
    bottom, right = ink.max(axis=0)
    return (top >= margin and left >= margin
            and bottom < IMG_SIZE - margin and right < IMG_SIZE - margin)


def _draw_digit(digit, rotate, vary_size, vary_position, min_size, max_size, max_angle, base_size):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)  # Black background
    draw = ImageDraw.Draw(img)

    # Font size
    font_size = random.randint(min_size, max_size) if vary_size else base_size

    font = load_font(font_size)

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position
    if vary_position:
        x = random.randint(0, max(0, IMG_SIZE - text_w))
        y = random.randint(0, max(0, IMG_SIZE - text_h))
    else:
        x = (IMG_SIZE - text_w) // 2
        y = (IMG_SIZE - text_h) // 2

    # bbox[0], bbox[1] is the offset between the text origin and the visible digit
    draw.text((x - bbox[0], y - bbox[1]), text, fill=255, font=font)

    # Rotation
    if rotate:
        angle = random.randint(-max_angle, max_angle)
        img = img.rotate(angle)

    return img

# Generate images
samples_per_digit = 10  # Adjust as needed

parser = argparse.ArgumentParser(description="Generate digit image datasets.")
parser.add_argument("folders", nargs="*", choices=list(folders),
                    help="variants to generate (default: all)")
parser.add_argument("--out", default=base_dir, help="output base directory")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--missing-only", action="store_true",
                    help="only create images whose files do not exist yet")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

for folder_name, params in folders.items():
    if args.folders and folder_name not in args.folders:
        continue
    for digit in range(10):
        folder_path = os.path.join(args.out, folder_name, str(digit))
        os.makedirs(folder_path, exist_ok=True)

        for i in range(samples_per_digit):
            file_path = os.path.join(folder_path, f"{digit}_{i}.png")
            if args.missing_only and os.path.exists(file_path):
                continue
            img = generate_image(digit, **params)
            img.save(file_path)

print("✅ Done! Images saved to:", args.out)