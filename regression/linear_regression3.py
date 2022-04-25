import numpy as np
import pandas as pd
from tqdm import trange

from my_dataset_dataloader import MyDataloader

def _stand(q): return (q - np.mean(q)) / (np.std(q) + 1e-7), np.mean(q), np.std(q)

def get_data(csv_file="prices.csv"):
    # all_datas = np.array([i.strip().split(',') for i in open(
    #     csv_file, 'r', encoding='utf-8')][1:], dtype=np.float32)
    all_datas = np.array(pd.read_csv(csv_file), dtype=np.float32)
    x, y = all_datas[...,1:], all_datas[..., 0]
    x, mean_x, std_x = _stand(x)  #标准化
    y, mean_y, std_y = _stand(y)  #标准化
    all_datas = np.hstack([x, y[:, None]])
    return all_datas, mean_x, std_x, mean_y, std_y

def main():
    all_datas, mean_x, std_x, mean_y, std_y = get_data()

    k = np.ones((6, 1))
    b = 1
    lr = 0.01
    batch_size = 10
    epoch = 10000
    # epoch = 10


    dataloader = MyDataloader(all_datas, batch_size, shuffle=True)
    for _ in trange(epoch):
        for batch_data in dataloader:
            # 必须要这步[:,None]，label shape(10)==>(10,1)
            x, label = batch_data[..., :-1], batch_data[..., -1][:,None]
            pre = x @ k + b
            loss = np.mean(((pre - label) ** 2) / batch_size)
            G = (pre - label) / batch_size
            delta_k = x.T @ G
            delta_b = np.mean(G)

            k += -delta_k * lr
            b += -delta_b * lr
    print(f"k: {k}, b: {b}")
    while True:
        # 室,厅,卫,面积（平米）,楼层,建成年份
        bedroom = float(input("请输入室: "))
        ting = float(input("请输入厅: "))
        wei = float(input("请输入卫: "))
        area = float(input("请输入面积: "))
        floor = float(input("请输入楼层: "))
        year = float(input("请输入建成年份: "))
        val_x = (np.array([bedroom, ting, wei, area, floor, year]) - mean_x) / std_x
        print(f"预测房价: {(val_x @ k + b) * std_y + mean_y}")


if __name__ == '__main__':
    main()