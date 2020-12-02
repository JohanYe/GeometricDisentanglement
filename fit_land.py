import torch
import torchvision.datasets
import model
from utils import load_checkpoint, custom_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import land
import seaborn as sns
from geoml import manifold, curve
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_style("darkgrid")
model_folder = "./model/best_beta1/"
load_land = False

batch_size = 512
layers = torch.linspace(28 ** 2, 2, 3).int()
num_components = 50
label_thresh = 4  # include only a subset of MNIST classes
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

## Data
mnist_train = torchvision.datasets.MNIST('data/', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float() / 255
y_train = mnist_train.targets
idx = y_train < label_thresh  # only use digits 0, 1, 2, ...
num_classes = y_train[idx].unique().numel()
x_train = x_train[idx]
y_train = y_train[idx]
N = x_train.shape[0]

# load model
net = model.VAE(x_train, layers, num_components=num_components, device=device)
ckpt = load_checkpoint(model_folder + 'best.pth.tar', net)
net.init_std(x_train, gmm_mu=ckpt['gmm_means'], gmm_cv=ckpt['gmm_cv'], weights=ckpt['weights'])
saved_dict = ckpt['state_dict']
new_dict = net.state_dict()
new_dict.update(saved_dict)
net.load_state_dict(new_dict)
net.eval()

with torch.no_grad():
    z = torch.chunk(net.encoder(x_train.to(device)), chunks=2, dim=-1)[0].cpu()
    minz, _ = z.min(dim=0)  # d
    maxz, _ = z.max(dim=0)  # d
    z_data = torch.utils.data.TensorDataset(z)
    train_loader = torch.utils.data.DataLoader(z_data, batch_size=batch_size, shuffle=True)

if load_land:
    mu = torch.Tensor(np.load(model_folder + 'land_mu.npy')).to(device).requires_grad_(True)
    std = torch.Tensor(np.load(model_folder + 'land_std.npy')).to(device).requires_grad_(True)
else:  # manual init
    mu_np = np.expand_dims(np.array([0, 0]), axis=0)
    mu = torch.tensor(mu_np).to(device).float().requires_grad_(True)
    std = torch.tensor([[1 / 30, 1 / 60], [1 / 61, 1 / 31]]).to(device).float().requires_grad_(True)

# meshgrid creating
meshsize = 60
ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize)
ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize)
Mx, My = torch.meshgrid(ran0, ran1)
Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
Mxy.requires_grad = False
dv = (ran0[-1] - ran0[0]) * (ran1[-1] - ran1[0]) / (meshsize ** 2)
batch_size = 1024
iters = (Mxy.shape[0] // batch_size) + 1
curves = {}

optimizer = torch.optim.Adam([mu], lr=1e-2, weight_decay=1e-2)
Cs, mus, lpzs, constants, distances = [], [], [], [], []
lpzs_log, mu_log, constant_log, distance_log = {}, {}, {}, {}
n_epochs = 1

net.eval()
for epoch in range(1, n_epochs + 1):
    for idx, batch in enumerate(tqdm(train_loader)):
        # data
        lpz, init_curve, dist2, constant = land.LAND_fullcov(loc=mu,
                                                             A=std,
                                                             z_points=batch[0].to(device),
                                                             dv=dv,
                                                             grid_points=Mxy,
                                                             model=net)
        lpz.mean().backward()
        optimizer.step()

        mus.append(mu.cpu().detach())
        lpzs.append(lpz)
        constants.append(constant.unsqueeze(0))
        distances.append(dist2.sqrt().unsqueeze(0))

    lpzs_log[epoch] = torch.cat(lpzs).mean().item()
    mu_log[epoch] = torch.cat(mus).mean().item()

    print('Epoch: {}, P(z): {:.4f}, mu: [{:.4f},{:.4f}], std: {}'.format(epoch,
                                                                         torch.cat(lpzs).mean().item(),
                                                                         torch.cat(mus).mean(dim=0)[0].item(),
                                                                         torch.cat(mus).mean(dim=0)[0].item(),
                                                                         np.round(std.data.tolist(), 4)))
