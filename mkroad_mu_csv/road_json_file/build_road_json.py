import os
import json
import argparse
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def gather_images(input_dir: Path, recursive: bool):
    files = []
    if recursive:
        for p in input_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
    else:
        for p in input_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
    return sorted(files)


def main():
    #输入文件夹
    #input_dir = "D:/dataset/RSCD/vali_20k"
    input_dir = "selected_images"
    #输出的路面json文件
    output_path = "road_info/road.json"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',default=input_dir ,help='Folder containing images')
    parser.add_argument('--output', '-o', default=output_path, help='Output json path (default: road.json)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively search subfolders')
    parser.add_argument('--relative', '-R', default=None,
                        help='Make paths relative to this base directory (default: current working dir)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    base = Path(args.relative) if args.relative is not None else Path.cwd()

    images = gather_images(input_dir, args.recursive)
    if not images:
        print('No images found in', input_dir)

    # convert to relative or absolute strings
    out_list = []
    for p in images:
        try:
            rel = p.relative_to(base)
            out_list.append(str(rel).replace('\\', '/'))
        except Exception:
            # fallback to absolute
            out_list.append(str(p).replace('\\', '/'))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_list, f, indent=4, ensure_ascii=False)

    print(f'Wrote {len(out_list)} image paths to {output_path}')


if __name__ == '__main__':
    main()
