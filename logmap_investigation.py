# import matplotlib
# matplotlib.use('TkAgg')
import torch
import torchvision.datasets
import model
from utils import load_checkpoint, custom_dataset, print_stdout
import visualize
import numpy as np
import land
import seaborn as sns
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

# vae5-3, weird direction
torch.manual_seed(0)
sns.set_style("darkgrid")
model_folder = "./model/mnist2-3/"
model_name = 'init_1_sampled'
model_dir = model_folder + model_name + "/"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

mu = torch.from_numpy(np.load(model_dir + "LAND_mu_.npy")).to(device).requires_grad_(False).float()
A = torch.from_numpy(np.load(model_dir + "LAND_std_.npy")).to(device).requires_grad_(False).float()

batch_size = 1024
layers = torch.linspace(28 ** 2, 2, 3).int()
num_components = 50

## Data
mnist_train = torchvision.datasets.MNIST('data/', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float() / 255
y_train = mnist_train.targets

if "0" in model_folder:
    label_thresh = 2  # include only a subset of MNIST classes
    idx = y_train < label_thresh  # only use digits 0, 1, 2, ...
elif "vae" in model_folder:
    label_thresh = 4  # include only a subset of MNIST classes
    idx = y_train < label_thresh  # only use digits 0, 1, 2, ...
elif "mnist2" in model_folder:
    label_thresh = 2  # include only a subset of MNIST classes
    idx = y_train == label_thresh  # only use digits 0, 1, 2, ...
else:
    label_thresh = 1  # include only a subset of MNIST classes
    idx = y_train == label_thresh  # only use digits 0, 1, 2, ...

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
    z = torch.chunk(net.encoder(x_train.to(device)), chunks=2, dim=-1)[0]
z_data = custom_dataset(z)
z_loader = torch.utils.data.DataLoader(z_data, batch_size=batch_size, shuffle=False)
curve_endings = []
for i, batch in enumerate(z_loader):
    mu_repeated = mu.repeat([batch.shape[0], 1])
    with torch.no_grad():
        tmp, c, curve_ending = net.logmap(p0=mu_repeated, p1=batch, curve=None)
    if i == 0:
        logmapping = tmp.detach().cpu()
        logmap_curves = [c]
        curve_endings = curve_ending.detach().cpu()
    else:
        logmapping = torch.cat((logmapping, tmp.detach().cpu()), dim=0)
        curve_endings = torch.cat((curve_endings, curve_ending.detach().cpu()), dim=0)
        logmap_curves.append(c)
logmapping = logmapping.numpy().squeeze(1)

fig, ax = plt.subplots(figsize=(10, 10))
labels = y_train
for label in labels.unique():
    idx = labels == label
    logmapping_i = logmapping[idx]
    plt.plot(logmapping_i[:, 0], logmapping_i[:, 1], '.', markersize=0.7)

if "mnist1-2" in model_dir:
    corners = torch.Tensor([[-1.65, -0.38], [-2.16, -1.33], [-1.79, -2.96], [1, -2.35], [1.9, -1.47], [2.37, 0.37],
                            [1.37, 2.23],[-0.65, 2.57]])
elif "mnist1-3" in model_dir:
    corners = torch.Tensor([[-2.07, 1.59], [-2.09, -0.3], [-0.6, -1.8], [1.18, 0.17], [-0.44, 1.54]])
elif "mnist1-5" in model_dir:
    corners = torch.Tensor([[-0.75, -1.55], [-1.73, -1.69], [-1.26, -1.98], [0.5, -1.96], [1.43, -1.58], [3.22, -1.1],
                            [2.68, 0.6],[1.87, 1.37],[0.87,1.68]])
elif "mnist01-3" in model_dir:
    corners = torch.Tensor([[-1.6,2.34], [0, -2.1], [1.09, -2.8],[0.19,2.13],[-0.25, 2.13]])
elif "vae2" in model_dir:
    corners = torch.Tensor([[-1.26, 1.45], [-2.27, -0.75], [-0.91, -2.1],[-0.16,-2.6],[0.76, -2.3], [1.4,-2.96],
                            [1.35,-1.35],[2.7, -0.85],[1.75,1.79], [1.5, 2.55],[0.7,2.05],[0.4,3.23]])
elif "mnist2-3" in model_dir:
    corners = torch.Tensor([[-2.27,0.45],[-2.54,-1.75],[-2.3,-2.8],[-1.46,-3],[-1.33,2.3],[0.2,-1.96],[1.64,-0.94],[1.9,-0.11],
                            [1.97,1.12],[-0.14,2.23],[1,1.76],[-2.3,0.52]])
ax.plot(corners[:, 0], corners[:, 1], linestyle='', marker='o', markersize=5, c="black")
plt.title('Logmapping')
plt.savefig(model_dir + 'starfish.png', box_inches="tight")


for row in range(corners.shape[0]):
    diff = torch.Tensor(logmapping) - corners[row]
    min_diff_idx = diff.abs().sum(dim=1).min(dim=0)[1]
    curvelist_idx = int(min_diff_idx / batch_size)
    intra_curvelist_idx = min_diff_idx % batch_size
    # sanity checking if correct curve
    #print(logmap_curves[curvelist_idx][intra_curvelist_idx].end, curve_endings[min_diff_idx], diff.sum(dim=1).abs().min(dim=0)[0].data.tolist())
    curve = logmap_curves[curvelist_idx][intra_curvelist_idx]

    curve_points = curve(torch.linspace(0, 1, 10).to(device))
    p_xGz = net.decode(curve_points)
    n_imgs = curve_points.shape[0]

    fig, ax = plt.subplots(2, n_imgs, figsize=(3 * n_imgs, 7), constrained_layout=True)
    x_plot_mean = p_xGz.mean.cpu().detach().numpy().reshape(n_imgs, 28, 28)
    x_plot_noise = p_xGz.sample().cpu().detach().numpy().reshape(n_imgs, 28, 28)
    for i in range(n_imgs):
        ax[0, i].imshow(x_plot_mean[i], cmap="gray")
        ax[0, i].axis("off")
        ax[1, i].imshow(x_plot_noise[i], cmap="gray")
        ax[1, i].axis("off")
    plt.savefig(model_dir + 'logmap_traversal' + str(row) + '.png', bbox_inches='tight')
    plt.close()
