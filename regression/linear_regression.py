import numpy as np
from tqdm import trange

from my_dataset_dataloader import MyDataloader


def main():
    years = (np.array([i for i in range(2000, 2022)]) - 2000) / 22   # 年份 2000 ~ 2021
    prices = np.array(
        [10000, 11000, 12000, 13000, 14000, 12000, 13000, 16000, 18000, 20000, 19000, 22000, 24000, 23000, 26000, 35000,
         30000, 40000, 45000, 52000, 50000, 60000]) / 60000
    all_data = np.dstack([years, prices]).squeeze()

    k = 1
    b = 1
    lr = 0.01
    batch_size = 1
    epoch = 10000

    dataloader = MyDataloader(all_data, batch_size, shuffle=True)
    for _ in trange(epoch):
        for batch_data in dataloader:
            x, label = batch_data[..., 0], batch_data[..., 1]
            pre = k * x + b
            loss = (pre - label) ** 2
            delta_k = 2 * (pre - label) * x
            delta_b = 2 * (pre - label)
            

            k += -delta_k * lr
            b += -delta_b * lr
    print(f"k: {k}, b: {b}")
    while True:
        year = (float(input("请输入年份: ")) - 2000) / 22
        print(f"预测房价: {(k * year + b) * 60000}")


if __name__ == '__main__':
    main()