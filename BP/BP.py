import struct
import numpy as np
from matplotlib import pyplot as plt

from my_dataset_dataloader import MyDataloader

def sigmoid(q): return 1 / (1 + np.exp(-q))

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex,axis=1,keepdims=True)  # 多维度矩阵必须要keepdims=True
    return  ex / sum_ex

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



def main():
    # ----------加载数据-----------
    train_datas = load_images("../data/train-images.idx3-ubyte") / 255
    train_label = make_one_hot(load_labels("../data/train-labels.idx1-ubyte"))
    all_datas = np.hstack([train_datas, train_label])

    test_datas = load_images("../data/t10k-images.idx3-ubyte") / 255
    test_label = load_labels("../data/t10k-labels.idx1-ubyte")

    # ----------定义参数-----------
    w1 = np.random.normal(0, 1, size=(784, 256))
    b1 = 1
    w2 = np.random.normal(0, 1, size=(256, 10))
    b2 = 1

    lr = 0.1
    batch_size = 10
    epoch = 10

    # ----------训练----------
    dataloader = MyDataloader(all_datas, batch_size, shuffle=True)
    for _ in range(epoch):
        for batch_data in dataloader:
            x, label = batch_data[..., :-10], batch_data[..., -10:]

            # ----------前向推理----------
            sig_h = sigmoid(x @ w1 + b1)
            pre = softmax(sig_h @ w2 + b2)

            loss = -np.sum(label * np.log(pre))

            # ----------BP----------
            G2 = (pre - label) / batch_size
            delta_w2 = sig_h.T @ G2
            delta_b2 = np.mean(G2)
            G1 = (G2 @ w2.T) * (sig_h * (1 - sig_h)) / batch_size
            delta_w1 = x.T @ G1
            delta_b1 = np.mean(G1)

            w2 -= delta_w2 * lr
            b2 -= delta_b2 * lr
            w1 -= delta_w1 * lr
            b1 -= delta_b1 * lr

        # print(loss)

        # ----------预测----------
        val_l = softmax(sigmoid(test_datas @ w1 + b1) @ w2 + b2)
        val_l = np.argmax(val_l, axis=1)
        acc = np.sum(val_l == test_label) / len(test_label)
        print(acc)

if __name__ == '__main__':
    main()