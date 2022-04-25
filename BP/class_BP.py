import struct
import numpy as np
from matplotlib import pyplot as plt

from my_dataset_dataloader import MyDataloader


def sigmoid(q):
    return 1 / (1 + np.exp(-q))

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex,axis=1,keepdims=True)  # 多维度矩阵必须要keepdims=True
    return ex / sum_ex

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

class Linear:
    def __init__(self, in_num, out_num):
        self.weight = np.random.normal(0, 1, size=(in_num, out_num))
        self.u = 0.9
        self.vt = 0

    def forward(self, x):
        self.x = x
        return x @ self.weight

    def backward(self, G):
        delta_weight = (self.x.T @ G)
        delta_x = G @ delta_weight.T

        # # ----------SGD----------
        # self.weight -= lr * delta_weight

        # ----------Momentum SGD----------
        self.vt = self.vt * self.u - lr * delta_weight
        self.weight = self.weight + self.vt


        return delta_x

    def __call__(self, x):
        return self.forward(x)

class Sigmoid:
    def forward(self, x):
        self.x = sigmoid(x)
        return self.x

    def backward(self, G):
        return G * self.x * (1 - self.x)

    def __call__(self, x):
        return self.forward(x)

class Softmax:
    def forward(self, x):
        self.x = softmax(x)
        return self.x

    def backward(self, G):
        return (self.x - G) / self.x.shape[0]

    def __call__(self, x):
        return self.forward(x)

class MyModel:
    def __init__(self, layers):
        self.layers = layers

    # label必须放在这里，因为__call__只会调用一次forward，必须把所有参数都一次性放进来
    def forward(self, x, label=None):
        for layer in self.layers:
            x = layer(x)

        if label is not None:
            self.label = label
        return x

    # def backward(self, G):  # 此处把G去了
    def backward(self):
        G = self.label
        for layer in self.layers[::-1]:
            G = layer.backward(G)

    def __call__(self, *args):
        return self.forward(*args)

def main():
    # ----------加载数据-----------
    train_datas = load_images("../data/train-images.idx3-ubyte") / 255
    train_label = make_one_hot(load_labels("../data/train-labels.idx1-ubyte"))
    all_datas = np.hstack([train_datas, train_label])

    test_datas = load_images("../data/t10k-images.idx3-ubyte") / 255
    test_label = load_labels("../data/t10k-labels.idx1-ubyte")

    # ----------定义参数-----------
    model = MyModel([
        Linear(in_num=784, out_num=256),
        Sigmoid(),
        Linear(in_num=256, out_num=10),
        Softmax()
    ])

    global lr
    lr = 0.05
    batch_size = 200
    epoch = 10

    # ----------训练----------
    dataloader = MyDataloader(all_datas, batch_size, shuffle=True)
    for e in range(epoch):

        for batch_data in dataloader:
            x, label = batch_data[..., :-10], batch_data[..., -10:]

            # ----------前向推理----------
            x = model(x, label)
            loss = -np.sum(label * np.log(x))
            model.backward()

        # print(f"loss: {loss}")

        # ----------预测----------
        val_l = model(test_datas)
        val_l = np.argmax(val_l, axis=1)
        acc = np.sum(val_l == test_label) / len(test_label)
        print(f"epoch: {e}, acc: {acc}")

if __name__ == '__main__':
    main()