import os
import shutil
from PIL import Image

# Define source and destination folders
src_folder = "my_scene/sofa/images"
dst_folder = "my_scene/sofa_resize/input"
os.makedirs(dst_folder, exist_ok=True)

# Get a sorted list of all image files in the source folder
all_files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

# Select the first 100 images
sampled_files = all_files

# Process each image
for file in sampled_files:
    src_path = os.path.join(src_folder, file)
    dst_path = os.path.join(dst_folder, file)

    # Open the image
    img = Image.open(src_path)

    # Resize to half the original size (1/2 H × 1/2 W)
    new_size = (img.width // 4, img.height // 4)
    img_resized = img.resize(new_size, Image.ANTIALIAS)

    # Save resized image
    img_resized.save(dst_path)
    # img.save(dst_path)

print(f"Resized and copied {len(sampled_files)} images to {dst_folder}")
