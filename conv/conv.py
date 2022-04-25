import cv2
import glob
import math
import numpy as np
from matplotlib import pyplot as plt

def get_imgs(path):
    imgs = []
    for file in glob.glob(path+'/*'):
        img = cv2.imread(file)
        img = cv2.resize(img, (150, 150))
        img = img.transpose(2, 0, 1)
        imgs.append(img)
    return np.array(imgs)

def conv(imgs, kernel):
    padding, stride = 0, 1
    batch_size, img_channel, img_h, img_w = imgs.shape  # (4,3,150,150), img_channel==kernel_channel
    batch_kernel, kernel_channel, kernel_h, kernel_w = kernel.shape  # (2, 3, 3, 3)
    feature_map_h, feature_map_w = (img_h + 2 * padding - kernel_h) // stride + 1, \
                                   (img_w + 2 * padding - kernel_w) // stride + 1  # 148, 148

    linear_kernel = kernel.reshape(batch_kernel, -1)  # shape: (2, 27)

    linear_imgs = np.zeros((batch_size,
                            kernel_channel * kernel_h * kernel_w,
                            feature_map_h * feature_map_w
                            )) # shape: (4, 27, 21904)
    i = 0
    for h in range(feature_map_h):
        for w in range(feature_map_w):
            kernel_img = imgs[:, :, h: h+kernel_h, w: w+kernel_w]  # shape:(4, 3, 3, 3)
            kernel_img = kernel_img.reshape(batch_size, -1)  # shape: (4, 27)
            linear_imgs[:, :, i] = kernel_img  # linear_imgs[:, :, i].shape: (4, 27)
            i += 1
    # (2, 27) @ (4, 27, 21904) ==> (4, 2, 21904)
    C_out = linear_kernel @ linear_imgs
    return C_out




if __name__ == '__main__':
    # kernel.shape: (2, 3, 3, 3)
    kernel = np.array([
        [
            [
                [-1, -2, -3],
                [-1, -2, -3],
                [-1, -10, 1]
            ],
            [
                [0, 3, 3],
                [-1, -2, -3],
                [1, 1, 1]
            ],
            [
                [3, 3, 3],
                [-1, -9, 0],
                [-1, -2, -3]
            ]
        ],
        [
            [
                [1, -1, 0],
                [1, -1, 0],
                [1, -1, 0]
            ],
            [
                [1, -1, 0],
                [1, -1, 0],
                [1, -1, 0]
            ],
            [
                [1, -1, 0],
                [1, -1, 0],
                [1, -1, 0]
            ]
        ],

    ])

    img_path = "img"
    imgs = get_imgs(img_path)  # imgs.shape: (4,3,150,150)
    pre = conv(imgs, kernel)  # shape:(4, 2, 21904)

    batch, channel, feature_shape = pre.shape
    for i in range(batch):
        for j in range(channel):
            tmp_img = pre[i, j, :].reshape(-1, int(np.sqrt(feature_shape))) # shape: (148, 148)
            plt.imshow(tmp_img, cmap="gray")
            plt.show()


    print()