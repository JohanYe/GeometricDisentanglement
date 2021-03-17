import torch
import os
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms as transforms
import torch.distributions as td

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
    if experiment_parameters["dataset"] != "mnist":
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
