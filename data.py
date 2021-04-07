import torch
import os
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms as transforms
import torch.distributions as td
from os.path import join as join_path
import re

def load_data(experiment_parameters, root="./data"):
    if "outlier" in experiment_parameters["dataset"] or "augment" in experiment_parameters["dataset"]:
        data_path = join_path("./data", experiment_parameters["dataset"])
        x_train, y_train, N_train = load_mnist_augment(data_path)
        x_test = None
        print("data samples loaded", N_train)
    elif "bodies" in experiment_parameters["dataset"]:
        data_path = join_path("./data", "bodies/")
        x_train, y_train, N_train = load_bodies(data_path)
        x_test, y_test, N_test = None, None, None
    elif experiment_parameters["dataset"] == "mnist":
        x_train, y_train, N_train = load_mnist_train(root="./data", label_threshold=4, one_digit=False)
        x_test, y_test, N_test = load_mnist_train(root="./data", label_threshold=4, one_digit=False, train=False)
    elif experiment_parameters["dataset"] == "dsprites":
        path = join_path(root, "dsprites/data_split.pt")
        data = torch.load(path)
        x_train = data["train_set"].reshape(data["train_set"].shape[0],-1)
        x_test = data["test_set"].reshape(data["test_set"].shape[0], -1)
        N_train = x_train.shape[0]
        N_test = x_test.shape[0]
        y_train, y_test = torch.zeros(N_train), torch.zeros(N_test)
    else:
        exp_regex = re.findall(r"([a-zA-Z]+)(\d+)", experiment_parameters["dataset"])
        num_max = 0
        for i, c in enumerate(exp_regex[0][1]):
            if num_max < int(c):
                num_max = int(c)
            one_digit = False if i > 0 else True
        x_train, y_train, N_train = load_mnist_train(root="./data", label_threshold=num_max, one_digit=one_digit)
        x_test, y_test, N_test = load_mnist_train(root="./data", label_threshold=4, one_digit=False, train=False)
    return x_train, y_train, N_train, x_test, y_test, N_test


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
        z_loc, z_scale = torch.chunk(net.encoder(x_train.to(device)), chunks=2, dim=-1)  # [0] = mus
        z_loc = z_loc.cpu()
        z_scale = z_scale.cpu()
    z_data = sample_latents(z_loc, z_scale.mul(0.5).exp())
    if x_test is None:
        N_train = int(0.9 * len(z_data))
        N_test = len(z_data) - int(0.9 * len(z_data))
        train_set, test_set = torch.utils.data.random_split(z_data, [N_train, N_test])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader, N_train, N_test, z_loc
    else:
        with torch.no_grad():
            z_loc_test, z_scale_test = torch.chunk(net.encoder(x_test.to(device)), chunks=2, dim=-1)  # [0] = mus
            z_loc_test = z_loc_test.cpu()
            z_scale_test = z_scale_test.cpu()
        train_set = sample_latents(z_loc, z_scale.mul(0.5).exp())
        test_set = sample_latents(z_loc_test, z_scale_test.mul(0.5).exp())
        N_train = len(train_set)
        N_test = len(test_set)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader, N_train, N_test, z_loc

def data_split(x_train, x_test, batch_size):
    """ y is neglected due to y being solely for visualization """
    if x_test is None:
        N_train = int(0.9 * len(x_train))
        N_test = len(x_train) - int(0.9 * len(x_train))
        train_set, test_set = torch.utils.data.random_split(x_train, [N_train, N_test])
    else:
        train_set = torch.utils.data.TensorDataset(x_train)
        test_set = torch.utils.data.TensorDataset(x_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

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


class sample_latents(torch.utils.data.Dataset):
    def __init__(self, mu_tensor, lv_tensor, labels=None):
        self.mu_tensor = mu_tensor
        self.lv_tensor = lv_tensor
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None

    def __getitem__(self, index):
        if self.labels is None:
            return td.normal.Normal(loc=self.mu_tensor[index], scale=self.lv_tensor[index]).sample()
        else:
            return td.normal.Normal(loc=self.mu_tensor[index], scale=self.lv_tensor[index]).sample(), self.labels[index]

    def __len__(self):
        return self.mu_tensor.size(0)

class custom_dataset(torch.utils.data.Dataset):
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

