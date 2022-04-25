import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt

from my_dataset_dataloader import MyDataloader

def _stand(q): return (q - np.mean(q)) / (np.std(q) + 1e-7), np.mean(q), np.std(q)

def sigmoid(q): return 1 / (1 + np.exp(-q))

def plot_xy(k=np.array([[-0.39969241], [-0.23562142]]), b=4.120622218977447):
    ys = xs = np.array([i for i in range(15)])
    xs, ys = np.meshgrid(xs, ys)
    X = np.dstack((xs, ys))
    pred_y = sigmoid(X @ k + b).squeeze()
    cats = np.argwhere(pred_y > 0.5)
    dogs = np.argwhere(pred_y <= 0.5)

    plt.plot(cats[...,0], cats[...,1],
             color='limegreen',
             marker='.',
             linestyle='')
    plt.plot(dogs[...,0], dogs[...,1],
             color='red',
             marker='.',
             linestyle='')
    plt.grid(True)
    plt.show()

def main():
    dogs = np.array([[8.9, 12], [9, 11], [10, 13], [9.9, 11.2], [12.2, 10.1], [9.8, 13], [8.8, 11.2]],
                    dtype=np.float32)  # 0
    cats = np.array([[3, 4], [5, 6], [3.5, 5.5], [4.5, 5.1], [3.4, 4.1], [4.1, 5.2], [4.4, 4.4]], dtype=np.float32)  # 1
    labels = np.array([0] * 7 + [1] * 7)[:, None]
    all_datas = np.hstack((np.concatenate((dogs, cats)), labels))


    k = np.ones((2, 1))
    b = 1
    lr = 0.01
    batch_size = 2
    epoch = 1000
    # epoch = 10


    dataloader = MyDataloader(all_datas, batch_size, shuffle=True)
    for _ in trange(epoch):
        for batch_data in dataloader:
            # 必须要这步[:,None]，label shape(10)==>(10,1)
            x, label = batch_data[..., :-1], batch_data[..., -1][:,None]
            pre = sigmoid(x @ k + b)
            loss = -np.sum((label * np.log(pre) + (1-label) * np.log(1-pre)))
            print(loss)
            G = (pre - label) / batch_size
            delta_k = x.T @ G
            delta_b = np.mean(G)

            k += -delta_k * lr
            b += -delta_b * lr
    print(f"k: {k}, b: {b}")
    while True:
        # 室,厅,卫,面积（平米）,楼层,建成年份
        f1 = float(input("请输入毛发长: "))
        f2 = float(input("请输入腿长: "))
        val_x = np.array([f1, f2])
        pre_y = sigmoid(val_x @ k + b)
        if pre_y > 0.5:
            print(f"预测结果：猫")
        else:
            print(f"预测结果：狗")



if __name__ == '__main__':
    # main()
    plot_xy()