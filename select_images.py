import os
import shutil
import argparse
from pathlib import Path

def select_and_copy_images(input_dir, output_dir):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all image files (assuming jpg and png)
    image_extensions = ('.jpg', '.png', '.jpeg')
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    all_files = [os.path.join(input_dir, f) for f in all_files]

    # Select dry concrete smooth or dry asphalt smooth images (first 1500)
    dry_list = [f for f in all_files if os.path.basename(f).endswith('dry-concrete-smooth.jpg') or os.path.basename(f).endswith('dry-asphalt-smooth.jpg') or os.path.basename(f).endswith('dry-asphalt-slight.jpg')]
    selected_dry = sorted(dry_list, key=lambda x: os.path.basename(x))[:1492]

    # Select ice, fresh snow, or water mud images (first 1300)
    ice_list = [f for f in all_files if os.path.basename(f).endswith('ice.jpg') or os.path.basename(f).endswith('fresh-snow.jpg') or os.path.basename(f).endswith('water-mud.jpg')]
    selected_ice = sorted(ice_list, key=lambda x: os.path.basename(x))[:1308]

    # Combine selected images
    selected_images = selected_dry + selected_ice

    # Copy to output directory
    for img_path in selected_images:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        shutil.copy(img_path, output_path)
        print(f"Copied {filename} to {output_dir}")

    print(f"Dry images copied: {len(selected_dry)}")
    print(f"Ice/Snow/Mud images copied: {len(selected_ice)}")
    print(f"Total images copied: {len(selected_images)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select and copy specific images from input directory to output directory.')
    input_path = "/Volumes/U_disc/RSCD/test_50k"
    output_path = "/Users/lzj/github_project/transformer_for_road_identification/selected_images"
    parser.add_argument('--input', type=str, default=input_path, help='Input directory containing images')
    parser.add_argument('--output', type=str, default=output_path, help='Output directory to copy selected images')
    args = parser.parse_args()

    select_and_copy_images(args.input, args.output)