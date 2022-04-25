import struct
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from my_dataset_dataloader import MyDataloader

def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)

def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)

def make_one_hot(l):
    labels = np.zeros((len(l), 10))
    for i, n in enumerate(l):
        labels[i][n] = 1
    return labels

class Nets(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(784, 256),
            nn.Sigmoid(),
            nn.Linear(256, 10),
            # nn.Sigmoid(),
            # nn.Linear(10, 1),
        )
        # self.l1 = nn.Linear(784, 256)
        # self.l2 = nn.Linear(256, 10)
        # self.l3 = nn.Linear(10, 1)

    def forward(self, x):
        # x = nn.Sigmoid()(self.l1(x))
        # x = self.l2(x)
        # x = self.l3(x)
        return self.linear(x)


def main():
    # ----------加载数据-----------
    batch_size = 100
    device = "cuda"

    train_datas = (load_images("../data/train-images.idx3-ubyte") / 255).astype(np.float32)
    train_label = load_labels("../data/train-labels.idx1-ubyte")[..., None].astype(np.float32)
    all_datas = np.hstack([train_datas, train_label])

    test_datas = (load_images("../data/t10k-images.idx3-ubyte") / 255).astype(np.float32)
    test_label = load_labels("../data/t10k-labels.idx1-ubyte").astype(np.float32)
    test_datas, test_label = torch.from_numpy(test_datas).to(device), torch.from_numpy(test_label).to(device).long()

    # ----------训练----------
    dataloader = MyDataloader(all_datas, batch_size, shuffle=True)

    model = Nets().to(device)

    optimizer = optim.SGD(params=model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for _ in range(10):
        model.train()
        for batch_data in dataloader:
            x, label = batch_data[..., :-1], batch_data[..., -1]
            x, label = torch.from_numpy(x).to(device), torch.from_numpy(label).to(device).long()
            pre = model(x)
            loss = criterion(pre, label)
            loss.backward()
            optimizer.step()

        model.eval()

        val_l = model(test_datas)
        val_l = torch.argmax(val_l, axis=1)

        acc = torch.true_divide(torch.sum(val_l == test_label), len(test_label))
        print(float(acc))

if __name__ == '__main__':
    main()