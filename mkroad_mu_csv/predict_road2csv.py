import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms

import importlib.util
from pathlib import Path
from model import swin_tiny_patch4_window7_224 as create_model



def load_weights(weights_path, model, device):
    ckpt = torch.load(weights_path, map_location=device)
    # common checkpoint wrappers
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # try non-strict load to allow head mismatch
        print("Strict load failed, trying non-strict load (some keys may be skipped):", e)
        model.load_state_dict(state_dict, strict=False)


def predict_image(img_path, model, transform, device, class_indict):
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path).convert('RGB')
    x = transform(img)
    x = torch.unsqueeze(x, 0).to(device)

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(x)).cpu()
        prob = torch.softmax(output, dim=0)
        idx = torch.argmax(prob).item()
        return idx, prob[idx].item(), prob


def multi_predict2csv():
    #输入的路面json文件（需要先执行好build_road_json）
    road_json_path = "road_info/road.json"
    #模型选取
    model_weights_path = "weights/model-9.pth"
    #mu字典
    dir_mu_json_path = "mkroad_mu_csv/road_json_file/road_mu_dir.json"
    #输出csv
    output_csv_path = "road_info/predict_road_info/predict_road_mu.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str,default= road_json_path , help='path to json file containing image paths (list)')
    parser.add_argument('--weights', type=str,default=model_weights_path , help='model weights path')
    parser.add_argument('--num_classes', type=int, default=27)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mu-json', type=str, default=dir_mu_json_path,
                        help='path to road mu json mapping file')
    parser.add_argument('--out-csv', type=str, default=output_csv_path,
                        help='output csv path for predictions')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # transform
    img_size = args.img_size
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load class indices
    json_label_path = 'class_indices.json'
    assert os.path.exists(json_label_path), f"cannot find {json_label_path}"
    with open(json_label_path, 'r') as f:
        class_indict = json.load(f)

    # load image list json
    assert os.path.exists(args.json), f"cannot find {args.json}"
    with open(args.json, 'r') as f:
        img_list = json.load(f)

    # load mu mapping
    mu_json_path = args.mu_json
    if os.path.exists(mu_json_path):
        with open(mu_json_path, 'r', encoding='utf-8') as f:
            mu_list = json.load(f)
        # build name -> mu dict
        mu_map = {item.get('name'): item.get('mu') for item in mu_list}
    else:
        mu_map = {}

    # create model and load weights
    model = create_model(num_classes=args.num_classes).to(device)
    assert os.path.exists(args.weights), f"cannot find {args.weights}"
    load_weights(args.weights, model, device)

    # iterate, collect results and print
    results = []
    for i, img_path in enumerate(img_list, start=1):
        try:
            idx, p, full_prob = predict_image(img_path, model, transform, device, class_indict)
            label = class_indict.get(str(idx), str(idx))
            mu = mu_map.get(label)
            mu_str = '' if mu is None else mu
            results.append((i, mu_str, img_path))
        except Exception:
            results.append((i, '', img_path))

    # write CSV output
    out_csv = args.out_csv
    import csv
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'mu', 'path'])
        for row in results:
            writer.writerow(row)


if __name__ == '__main__':
    multi_predict2csv()
