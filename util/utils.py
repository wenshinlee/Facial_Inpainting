import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def imgshow(img, unnormalize=True):
    """
    显示图片
    imgshow(image)
    imgshow(mask, unnormalize=False)
    imgshow(segmap[:, 0:1, ...], unnormalize=False),segmap 每一个通道代表一个面部特征的mask
    :param img:输入图片格式的 B,C,H,W.其中C可以为 1
    :param unnormalize:去归一化
    :return:None
    """
    img = torchvision.utils.make_grid(img)
    if unnormalize:
        img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

