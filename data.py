import torch
import os
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms as transforms

def load_mnist_train(root, label_threshold, one_digit=False, download=False, train=True):
    mnist_train = torchvision.datasets.MNIST(root, train=train, download=download)
    x = mnist_train.data.reshape(-1, 784).float() / 255
    y = mnist_train.targets
    if one_digit:
        idx = y == label_threshold
    else:
        idx = y < label_threshold
    print("MNIST samples loaded:", idx.sum())
    x = x[idx]
    y = y[idx].float()
    N = x.shape[0]
    return torch.Tensor(x), torch.Tensor(y), N

def train_test_split_latents(net, experiment_parameters, x_train, x_test=None, batch_size=512, device="cuda"):
    with torch.no_grad():
        z = net.encode(x_train.to(device)).sample().cpu()  # [0] = mus
    z_data = torch.utils.data.TensorDataset(z)
    if experiment_parameters["dataset"] != "mnist":
        N_train = int(0.9 * len(z_data))
        N_test = len(z_data) - int(0.9 * len(z_data))
        train_set, test_set = torch.utils.data.random_split(z_data, [N_train, N_test])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader, N_train, N_test
    else:
        with torch.no_grad():
            z_test = net.encode(x_test.to(device)).sample().cpu()  # [0] = mus
        train_set = CustomTensorDataset(z)
        test_set = CustomTensorDataset(z_test)
        N_train = len(train_set)
        N_test = len(test_set)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader, N_train, N_test



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


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, labels=None):
        self.data_tensor = data_tensor
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None

    def __getitem__(self, index):
        if self.labels is None:
            return self.data_tensor[index]
        else:
            return self.data_tensor[index], self.labels[index]

    def __len__(self):
        return self.data_tensor.size(0)
