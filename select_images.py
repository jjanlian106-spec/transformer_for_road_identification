import os
import json
import shutil
import argparse
import random

# Load road mu data
road_json_path = 'mkroad_mu_csv/road_json_file/road_mu_dir.json'
with open(road_json_path, 'r') as f:
    data = json.load(f)

mu_dict = {item['name']: item['mu'] for item in data}

# Define high and low mu classes
high_names = [name for name, mu in mu_dict.items() if mu >= 0.9]
low_names = [name for name, mu in mu_dict.items() if mu <= 0.3]

print(f"High mu classes (>=0.9): {high_names}")
print(f"Low mu classes (<=0.3): {low_names}")

# Parse arguments
parser = argparse.ArgumentParser(description='Select 2800 images: 1500 high mu, 1300 low mu, grouped by class.')
input_dir = "/Volumes/U_DISC/RSCD/test_50k"
output_dir = "/Users/lzj/github_project/transformer_for_road_identification/selected_images"
parser.add_argument('--input_dir', default=input_dir, help='Input directory containing images')
parser.add_argument('--output_dir', default=output_dir, help='Output directory for selected images')
args = parser.parse_args()

# Get list of jpg files
files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.jpg')]
print(f"Found {len(files)} jpg files in input directory.")

# Group files by class
high_files = {name: [] for name in high_names}
low_files = {name: [] for name in low_names}

for file in files:
    parts = file.split('-')
    if len(parts) < 2:
        continue
    name_part = '-'.join(parts[1:])
    if '.' in name_part:
        name_part = name_part.rsplit('.', 1)[0]
    name = name_part.replace('-', '_')
    if name in high_files:
        high_files[name].append(file)
    elif name in low_files:
        low_files[name].append(file)

print(f"High mu files per class: {{k: len(v) for k,v in high_files.items()}}")
print(f"Low mu files per class: {{k: len(v) for k,v in low_files.items()}}")

# Calculate number per class
num_high = 1500
num_low = 1300
num_classes_high = len(high_names)
num_classes_low = len(low_names)

per_class_high = num_high // num_classes_high
remainder_high = num_high % num_classes_high
per_class_low = num_low // num_classes_low
remainder_low = num_low % num_classes_low

# Select high mu images
selected_high = []
for i, name in enumerate(high_names):
    files_list = high_files[name]
    num_to_select = per_class_high + (1 if i < remainder_high else 0)
    if len(files_list) < num_to_select:
        print(f"Warning: Only {len(files_list)} files for {name}, selecting all.")
        num_to_select = len(files_list)
    selected = random.sample(files_list, num_to_select)
    selected_high.extend(selected)

# Select low mu images
selected_low = []
for i, name in enumerate(low_names):
    files_list = low_files[name]
    num_to_select = per_class_low + (1 if i < remainder_low else 0)
    if len(files_list) < num_to_select:
        print(f"Warning: Only {len(files_list)} files for {name}, selecting all.")
        num_to_select = len(files_list)
    selected = random.sample(files_list, num_to_select)
    selected_low.extend(selected)

print(f"Selected {len(selected_high)} high mu images, {len(selected_low)} low mu images.")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Copy selected images with ordered names
all_selected = selected_high + selected_low
for i, file in enumerate(all_selected):
    src = os.path.join(args.input_dir, file)
    dst = os.path.join(args.output_dir, f"{i+1:04d}_{file}")
    shutil.copy(src, dst)

print(f"Copied {len(all_selected)} images to {args.output_dir}")
