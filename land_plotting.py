import torch
import torchvision.datasets
import model
from utils import load_checkpoint, custom_dataset
import numpy as np
import matplotlib.pyplot as plt
import land
import seaborn as sns
from tqdm import tqdm

sns.set_style("darkgrid")

model_folder = "./model/best_beta1/"
model_name = "simple_land"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
mu_np = np.expand_dims(np.array([0.1343, 0.0037]), axis=0)
mu = torch.tensor(mu_np).to(device).float().requires_grad_(False)
std = torch.tensor([[-0.0386, 0.0507], [0.0331, 0.0504]]).to(device).float().requires_grad_(False)

batch_size = 1024
layers = torch.linspace(28 ** 2, 2, 3).int()
num_components = 50
label_thresh = 4  # include only a subset of MNIST classes

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

subset = 4000
# plot latent variables
with torch.no_grad():
    z = torch.chunk(net.encoder(x_train[:subset].to(device)), chunks=2, dim=-1)[0]
labels = y_train[:subset]

# plt.subplots(figsize=(10, 10))
#
# if z.shape[1] == 2:
#     minz, _ = z.min(dim=0)  # d
#     maxz, _ = z.max(dim=0)  # d
#     alpha = 0.1 * (maxz - minz)  # + 10% to each edge
#     minz -= alpha  # d
#     maxz += alpha  # d
#
#     # meshgrid creating
#     meshsize = 100
#     ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize)
#     ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize)
#     Mx, My = torch.meshgrid(ran0, ran1)
#     Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
#     Mxy.requires_grad = False
#     dv = (ran0[-1] - ran0[0]) * (ran1[-1] - ran1[0]) / (meshsize ** 2)
#     batch_size = 256
#     iters = (Mxy.shape[0] // batch_size) + 1
#     curves = {}
#
#     with torch.no_grad():
#         varim = net.decoder_scale(Mxy.to(device)).pow(2).mean(dim=-1).reshape(meshsize, meshsize)
#         varim = varim.cpu().detach().numpy()
#     plt.imshow(varim, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
#     plt.colorbar()
#
#     for label in labels.unique():
#         idx = labels == label
#         zi = z[idx].cpu().detach().numpy()
#         plt.plot(zi[:, 0], zi[:, 1], '.')
# else:
#     raise Exception('Latent dimension not suitable for plotting')
#
# # compute the geodesics
# mu_repeated = mu.repeat([Mxy.shape[0], 1]).detach()
# C, success = net.connecting_geodesic(mu_repeated, Mxy.to(device))
# C.plot()
# plt.show()

# contour plot
minz, _ = z.min(dim=0)  # d
maxz, _ = z.max(dim=0)  # d
alpha = 0.1 * (maxz - minz)  # + 10% to each edge
minz -= alpha  # d
maxz += alpha  # d
meshsize = 150
ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize, device=device)
ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize, device=device)
Mx, My = torch.meshgrid(ran0, ran1)
Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1).to(device)  # (meshsize^2)x2
dv = (ran0[-1] - ran0[0]) * (ran1[-1] - ran1[0]) / (meshsize ** 2)
batch_size = 256
dataset = custom_dataset(data_tensor=Mxy)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Note shuffle false
pz_plt, approx_constant = None, None

with torch.no_grad():
    approx_constant, metrics = land.estimate_constant_full(mu=mu, A=std, grid=Mxy, dv=dv, model=net, batch_size=256,
                                                           sum=False)

# plot latent variables
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
subset = 6000
with torch.no_grad():
    z = torch.chunk(net.encoder(x_train[:subset].to(device)), chunks=2, dim=-1)[0]
labels = y_train[:subset]
for y, label in enumerate(labels.unique()):
    idx = labels == label
    zi = z[idx].cpu().detach().numpy()
    ax[0].plot(zi[:, 0], zi[:, 1], '.', label=y)

constant_tmp = (approx_constant.reshape(meshsize, meshsize)).cpu()
# constant_tmp[constant_tmp > 2] = 2
ax[0].imshow(constant_tmp, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
ax[0].set_title('LAND constant Contours Plot')
ax[0].legend(loc='best')
ax[0].axis('off')
im = ax[1].imshow(constant_tmp, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()),
                  origin='lower')
ax[1].set_title('LAND constant Contours Plot')
fig.colorbar(im)
plt.tight_layout(h_pad=1)
plt.show()

for j, gridpoints_batch in enumerate(tqdm(data_loader)):
    mu_repeated = mu.repeat([gridpoints_batch.shape[0], 1])

    with torch.no_grad():
        D2, _, _ = net.dist2_explicit(mu_repeated, gridpoints_batch, A=std)
        exponential_term = (-1 * D2 / 2).squeeze(-1).exp()
        pz = (1 / approx_constant.sum()) * exponential_term

    if torch.isnan(D2.mean()):
        break

    if j == 0:
        pz_plt = pz.cpu().detach()
        distances = D2.cpu().detach()
    else:
        pz_plt = torch.cat((pz_plt, pz.cpu().detach()), dim=0)
        distances = torch.cat((distances, D2.cpu().detach()), dim=0)

# plot latent variables
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
subset = 6000
with torch.no_grad():
    z = torch.chunk(net.encoder(x_train[:subset].to(device)), chunks=2, dim=-1)[0]
labels = y_train[:subset]
for y, label in enumerate(labels.unique()):
    idx = labels == label
    zi = z[idx].cpu().detach().numpy()
    ax[0].plot(zi[:, 0], zi[:, 1], '.', label=y)

pz_plt_tmp = pz_plt.reshape(meshsize, meshsize)
ax[0].imshow(pz_plt_tmp, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
ax[0].set_axis_off()
ax[0].legend(loc='best')
im = ax[1].imshow(pz_plt_tmp, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
fig.colorbar(im)
ax[0].set_title('LAND p(z) Contours Plot')
ax[1].set_title('LAND p(z) Contours Plot')
plt.show()

# plot latent variables
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
with torch.no_grad():
    z = torch.chunk(net.encoder(x_train[:subset].to(device)), chunks=2, dim=-1)[0]
labels = y_train[:subset]
for y, label in enumerate(labels.unique()):
    idx = labels == label
    zi = z[idx].cpu().detach().numpy()
    ax[0].plot(zi[:, 0], zi[:, 1], '.', label=y)

distance_plot = distances.reshape(meshsize, meshsize)
ax[0].imshow(distance_plot, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
ax[0].set_axis_off()
ax[0].legend(loc='best')
im = ax[1].imshow(distance_plot, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()),
                  origin='lower')
fig.colorbar(im)
ax[0].set_title('LAND distances Plot')
ax[1].set_title('LAND distances Plot')

plt.show()
