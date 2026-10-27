import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# Output base directory
base_dir = "generated_digits"

# Folder structure
folders = {
    "centered": {"rotate": False, "vary_size": False, "vary_position": False},
    "different_sizes": {"rotate": False, "vary_size": True,  "vary_position": False},
    "different_rotations": {"rotate": True,  "vary_size": False, "vary_position": False},
}

IMG_SIZE = 28

def generate_image(digit, rotate=False, vary_size=False, vary_position=False):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)  # Black background
    draw = ImageDraw.Draw(img)

    # Font size
    font_size = random.randint(8, 18) if vary_size else 14

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

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

    draw.text((x, y), text, fill=255, font=font)

    # Rotation
    if rotate:
        angle = random.randint(-45, 45)
        img = img.rotate(angle)

    return img

# Generate images
samples_per_digit = 10  # Adjust as needed

for folder_name, params in folders.items():
    for digit in range(10):
        folder_path = os.path.join(base_dir, folder_name, str(digit))
        os.makedirs(folder_path, exist_ok=True)

        for i in range(samples_per_digit):
            img = generate_image(digit, **params)
            img.save(os.path.join(folder_path, f"{digit}_{i}.png"))

print("✅ Done! Images saved to:", base_dir)