import os
import json
import argparse
import sys

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from utils import read_split_data
from my_dataset import MyDataSet
import importlib.util
from pathlib import Path
# dynamic import of model.py (works when running from subfolders)
model_path = Path(__file__).resolve().parent / 'model.py'
if not model_path.exists():
    model_path = Path(__file__).resolve().parents[1] / 'model.py'
spec = importlib.util.spec_from_file_location("model_from_path", str(model_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
create_model = module.swin_tiny_patch4_window7_224


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)

        # Normalize by true label (columns) so color shows proportions per true class
        with np.errstate(all='ignore'):
            row_sums = matrix.sum(axis=1, keepdims=True)
            norm_matrix = np.divide(matrix, row_sums, where=row_sums != 0)

        # choose figure size depending on number of classes
        figsize_x = max(8, self.num_classes * 0.6)
        figsize_y = max(6, self.num_classes * 0.45)
        plt.figure(figsize=(figsize_x, figsize_y))

        im = plt.imshow(norm_matrix, cmap='Blues', vmin=0.0, vmax=1.0, interpolation='nearest', aspect='auto')

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45, fontsize=10)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels, fontsize=10)
        # 显示colorbar，表示归一化比例
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Proportion', rotation=270, labelpad=12)
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (rows normalized)')

        # 在图中标注数量/概率信息：显示原始计数 + (归一化比例)
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                count = int(matrix[y, x])
                prop = norm_matrix[y, x]
                text = f"{count}\n{prop:.2f}"
                # use white text for dark cells
                text_color = "white" if prop > 0.5 else "black"
                plt.text(x, y, text,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color=text_color,
                         fontsize=8)

        plt.tight_layout()
        plt.show()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, _, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 240
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes)
    # load pretrain weights
    assert os.path.exists(args.weights), "cannot find {} file".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    model.eval()
    with torch.no_grad():
        for val_data in tqdm(val_loader, file=sys.stdout):
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=27)
    parser.add_argument('--batch-size', type=int, default=8)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="D:/dataset/RSCD/train")

    # 训练权重路径
    parser.add_argument('--weights', type=str, default='weights/model-9.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
