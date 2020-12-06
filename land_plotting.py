import torch
import torchvision
import model
from utils import load_checkpoint, custom_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
sns.set_style("darkgrid")
model_folder = "./model/best_beta1/"
model_name = 'best_beta1'
mu_np = np.expand_dims(np.array([-0.2448, -0.0057]), axis=0)
mu = torch.tensor(mu_np).to(device).float().requires_grad_(True)
std = torch.tensor([[0.0214, 0.027], [-0.001, -0.0189]]).to(device).float().requires_grad_(True)

# batch_size = 1024 if hpc else 512
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

subset = 3000

# plot latent variables
with torch.no_grad():
    z = torch.chunk(net.encoder(x_train[:subset].to(device)), chunks=2, dim=-1)[0]
labels = y_train[:subset]

plt.subplots(figsize=(10, 10))

if z.shape[1] == 2:
    minz, _ = z.min(dim=0)  # d
    maxz, _ = z.max(dim=0)  # d
    alpha = 0.1 * (maxz - minz)  # d
    minz -= alpha  # d
    maxz += alpha  # d

    # meshgrid creating
    meshsize = 40
    ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize)
    ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize)
    Mx, My = torch.meshgrid(ran0, ran1)
    Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
    Mxy.requires_grad = False
    dv = (ran0[-1] - ran0[0]) * (ran1[-1] - ran1[0]) / (meshsize ** 2)
    batch_size = 256
    iters = (Mxy.shape[0] // batch_size) + 1
    curves = {}

    with torch.no_grad():
        varim = net.decoder_scale(Mxy.to(device)).pow(2).mean(dim=-1).reshape(meshsize, meshsize)
        varim = varim.cpu().detach().numpy()
    plt.imshow(varim, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
    plt.colorbar()

    for label in labels.unique():
        idx = labels == label
        zi = z[idx].cpu().detach().numpy()
        plt.plot(zi[:, 0], zi[:, 1], '.')
else:
    raise Exception('Latent dimension not suitable for plotting')

# compute the geodesics
mu_repeated = mu.repeat([Mxy.shape[0], 1]).detach()
C, success = net.connecting_geodesic(mu_repeated, Mxy.to(device))
C.plot()
plt.show()

# contour plot
meshsize = 75
ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize, device=device)
ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize, device=device)
Mx, My = torch.meshgrid(ran0, ran1)
Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1).to(device)  # (meshsize^2)x2
dv = (ran0[-1] - ran0[0]) * (ran1[-1] - ran1[0]) / (meshsize ** 2)
batch_size = 256
dataset = custom_dataset(endpoint_tensor=Mxy)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Note shuffle false
pz_plt, approx_constant = None, None

for j, (Mxy_batch, batch_index) in enumerate(tqdm(data_loader)):
    mu_repeated = mu.repeat([Mxy_batch.shape[0], 1])

    with torch.no_grad():
        D2, init_curve, success = net.dist2_explicit(mu_repeated, Mxy_batch, A=std)
        inside = (-(D2 + 1e-10) / 2).squeeze(-1)
        m = net.metric(Mxy_batch).det().sqrt()
        constant = (m * inside.exp()) * dv
        if approx_constant is None:
            approx_constant = constant.cpu()
        else:
            approx_constant = torch.cat((approx_constant, constant.cpu()), dim=0)

approx_constant_mean = approx_constant.sum()

for j, (Mxy_batch, batch_index) in enumerate(tqdm(data_loader)):
    mu_repeated = mu.repeat([Mxy_batch.shape[0], 1])

    with torch.no_grad():
        D2, init_curve, success = net.dist2_explicit(mu_repeated, Mxy_batch, A=std)
        inside = -(D2 + 1e-10) / 2
        m = net.metric(Mxy_batch).det().sqrt()
        pz = inside.exp() / approx_constant_mean
    if pz_plt is None:
        pz_plt = pz.cpu()
    else:
        pz_plt = torch.cat((pz_plt, pz.cpu()), dim=0)

# plot latent variables
fig, ax = plt.subplots(figsize=(7, 7))
subset = 2000
with torch.no_grad():
    z = torch.chunk(net.encoder(x_train[:subset].to(device)), chunks=2, dim=-1)[0]
labels = y_train[:subset]
for label in labels.unique():
    idx = labels == label
    zi = z[idx].cpu().detach().numpy()
    plt.plot(zi[:, 0], zi[:, 1], '.')

constant_tmp = (approx_constant.reshape(meshsize, meshsize))
# constant_tmp[constant_tmp > 2] = 2
im = ax.imshow(constant_tmp, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
fig.colorbar(im)
plt.axis('off')
ax.set_title('LAND constant Contours Plot')
plt.tight_layout(h_pad=1)
plt.show()

# plot latent variables
fig, ax = plt.subplots(figsize=(10, 10))
subset = 3000
with torch.no_grad():
    z = torch.chunk(net.encoder(x_train[:subset].to(device)), chunks=2, dim=-1)[0]
labels = y_train[:subset]
for label in labels.unique():
    idx = labels == label
    zi = z[idx].cpu().detach().numpy()
    plt.plot(zi[:, 0], zi[:, 1], '.')

pz_plt_tmp = pz_plt.reshape(meshsize, meshsize)
# pz_plt_tmp[pz_plt_tmp < 0.00011] = 0.00011
im = ax.imshow(pz_plt_tmp, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
fig.colorbar(im)
ax.set_title('LAND p(z) Contours Plot')
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
normaldist = torch.distributions.MultivariateNormal(loc=torch.Tensor([0, 0]),
                                                    covariance_matrix=torch.Tensor([[5, 0], [0, 5]]))
normaldist_plt = normaldist.log_prob(Mxy.cpu()).exp()
normaldist_plt = normaldist_plt.reshape(75, 75)
im = ax.imshow(normaldist_plt, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()),
               origin='lower')
fig.colorbar(im)
ax.set_title('LAND distances Plot')
ax.grid(False)
plt.show()

# contour plot
meshsize = 100
ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize, device=device)
ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize, device=device)
Mx, My = torch.meshgrid(ran0, ran1)
Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1).to(device)  # (meshsize^2)x2
batch_size = 250
dataset = custom_dataset(endpoint_tensor=Mxy)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Note shuffle false
distances = None

with torch.no_grad():
    for j, (Mxy_batch, batch_index) in enumerate(tqdm(data_loader)):
        mu_repeated = mu.repeat([Mxy_batch.shape[0], 1])

        with torch.no_grad():
            D2, _, _ = net.dist2_explicit(mu_repeated, Mxy_batch, A=std)
        if distances is None:
            distances = D2.sqrt().cpu()
        else:
            distances = torch.cat((distances, D2.sqrt().cpu()), dim=0)

# plot latent variables
fig, ax = plt.subplots(figsize=(10, 10))
with torch.no_grad():
    z = torch.chunk(net.encoder(x_train[:subset].to(device)), chunks=2, dim=-1)[0]
labels = y_train[:subset]
for label in labels.unique():
    idx = labels == label
    zi = z[idx].cpu().detach().numpy()
    plt.plot(zi[:, 0], zi[:, 1], '.')

distance_plot = distances.reshape(meshsize, meshsize)
im = ax.imshow(distance_plot, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
fig.colorbar(im)
ax.set_title('LAND distances Plot')
ax.grid(False)
plt.show()
