import torch
import os
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms as transforms

def load_mnist_train(root, label_threshold, one_digit=False, download=False):
    mnist_train = torchvision.datasets.MNIST(root, train=True, download=download)
    x_train = mnist_train.data.reshape(-1, 784).float() / 255
    y_train = mnist_train.targets
    if one_digit:
        idx = y_train == label_threshold
    else:
        idx = y_train < label_threshold
    print("MNIST samples loaded:", idx.sum())
    x_train = x_train[idx]
    y_train = y_train[idx]
    N = x_train.shape[0]
    return torch.Tensor(x_train), torch.Tensor(y_train), N


def load_mnist_augment(data_path):
    load = np.load(data_path, allow_pickle=True)
    x_train = load['x_train']
    y_train = load['y_train']
    N = x_train.shape[0]
    return torch.Tensor(x_train), torch.Tensor(y_train), N


def load_body(filename, resize):
    im = Image.open(filename)
    im = im.resize((resize, resize))
    return transforms.ToTensor()(im)


def load_bodies(data_path):
    body_imgs = [x for x in os.listdir(data_path) if ".png" in x]
    x_train = torch.zeros([len(body_imgs), 64, 64])
    for i, file in enumerate(body_imgs):
        x_train[i] = load_body(data_path + file, x_train.shape[-1])
    N = x_train.shape[0]
    x_train = x_train.reshape(N, -1)
    y_train = torch.ones(N)
    return x_train, y_train, N

