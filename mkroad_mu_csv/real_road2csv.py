import json
import csv
from pathlib import Path
import argparse


def normalize_label_from_filename(path_str: str) -> str:
    """
    Extract label from filename and normalize to underscore-separated lower-case form.
    Example: '202201252343338-dry-asphalt-smooth.jpg' -> 'dry_asphalt_smooth'
    """
    p = Path(path_str)
    name = p.name
    stem = p.stem  # filename without suffix
    # If filename contains a separator between prefix and label, split once
    if '-' in stem:
        # split at first hyphen to preserve hyphens inside label
        parts = stem.split('-', 1)
        label_part = parts[1]
    else:
        label_part = stem

    # normalize: lower-case and replace hyphens with underscores
    label = label_part.lower().replace('-', '_')
    # also strip whitespace
    label = label.strip()
    return label


def build_mu_map(mu_json_path: Path) -> dict:
    if not mu_json_path.exists():
        raise FileNotFoundError(f"mu json not found: {mu_json_path}")
    with open(mu_json_path, 'r', encoding='utf-8') as f:
        mu_list = json.load(f)
    mu_map = {}
    for item in mu_list:
        name = item.get('name')
        mu = item.get('mu')
        if name is None:
            continue
        mu_map[name.lower()] = mu
    return mu_map


def real_road_mu2csv():
    parser = argparse.ArgumentParser()
    #输入实际路面对应的json文件（road_info文件夹下）
    road_json_path = "road_info/road.json"
    #输入mu映射json字典(不要动)
    mu_json_path = "mkroad_mu_csv/road_json_file/road_mu_dir.json"
    #输出的路面csv文件（road_info/real_road_info下）
    output_csv_path = "road_info/real_road_info/real_road_mu.csv"
    parser.add_argument('--road_json', default=road_json_path )
    parser.add_argument('--mu-json', default=mu_json_path )
    parser.add_argument('--output', '-o', default=output_csv_path, help='output CSV file')
    args = parser.parse_args()

    road_json_path = Path(args.road_json)
    mu_json_path = Path(args.mu_json)

    if not road_json_path.exists():
        raise SystemExit(f"road.json not found: {road_json_path}")

    mu_map = build_mu_map(mu_json_path)

    with open(road_json_path, 'r', encoding='utf-8') as f:
        paths = json.load(f)

    rows = []
    for i, p in enumerate(paths, start=1):
        label_norm = normalize_label_from_filename(p)
        mu = mu_map.get(label_norm)
        # try fallback: some names may use '_' vs '-' differences reversed
        if mu is None:
            # try replace underscores with hyphens and back to underscores
            alt = label_norm.replace('-', '_').replace('__', '_')
            mu = mu_map.get(alt)
        if mu is None:
            # last resort: try simple substring match
            for name_key in mu_map.keys():
                if label_norm in name_key or name_key in label_norm:
                    mu = mu_map[name_key]
                    break

        rows.append((i, '' if mu is None else mu, p))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'mu', 'path'])
        for r in rows:
            writer.writerow(r)

    print(f'Wrote {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    real_road_mu2csv()
