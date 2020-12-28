import torch
import torchvision.datasets
import model
from utils import load_checkpoint, custom_dataset, print_stderr
import visualize
import numpy as np
import land
import seaborn as sns
from tqdm import tqdm
import time

sns.set_style("darkgrid")
model_folder = "./model/best_beta1/"
model_name = 'best_beta1'
load_land = False
sampled = True
hpc = True
fast_train = False
debug_mode = False
full_cov = True
start_time = time.time()

batch_size = 256 if hpc else 64
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
    z = torch.chunk(net.encoder(x_train.to(device)), chunks=2, dim=-1)[0].cpu()  # [0] = mus
    minz, _ = z.min(dim=0)  # d
    maxz, _ = z.max(dim=0)  # d
    alpha = 0.1 * (maxz - minz)  # d
    minz -= alpha  # d
    maxz += alpha
    z_data = torch.utils.data.TensorDataset(z)
    train_loader = torch.utils.data.DataLoader(z_data, batch_size=batch_size, shuffle=True)

if load_land:
    mu = torch.Tensor(np.load(model_folder + 'land_mu.npy')).to(device).requires_grad_(True)
    std = torch.Tensor(np.load(model_folder + 'land_std.npy')).to(device).requires_grad_(True)
else:  # manual init
    mu_np = np.expand_dims(np.array([0, 0]), axis=0)
    mu = torch.tensor(mu_np).to(device).float().requires_grad_(True)
    if full_cov:
        std = torch.tensor([[1/100, 1/50], [1/50, 1/100]]).to(device).float().requires_grad_(True)
    else:
        std = torch.tensor([40.]).to(device).float().requires_grad_(True)

# meshgrid creating
meshsize = 100
ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize)
ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize)
Mx, My = torch.meshgrid(ran0, ran1)
Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
Mxy.requires_grad = False
dv = (ran0[-1] - ran0[0]) * (ran1[-1] - ran1[0]) / (meshsize ** 2)
iters = (Mxy.shape[0] // batch_size) + 1
curves = {}

if sampled:
    grid_prob, grid_metric, grid_metric_sum = land.LAND_grid_prob(grid=Mxy,
                                                                  model=net,
                                                                  batch_size=1024,
                                                                  device=device)

optimizer_mu = torch.optim.Adam([mu], lr=3e-4)
lpzs_log, mu_log, constant_log, distance_log = {}, {}, {}, {}
optimizer_std = torch.optim.Adam([std], lr=3e-4)
lpzs_log_std, std_log, constant_log_std, distance_log_std = {}, {}, {}, {}
n_epochs = 2 if debug_mode else 10
total_epochs = 0

net.eval()
for j in range(20):
    if j != 0:
        for epoch in range(total_epochs + 1,total_epochs +  n_epochs + 1):
            total_epochs += 1
            Cs, mus, lpzs, constants, distances = [], [], [], [], []
            for idx, batch in enumerate(tqdm(train_loader)):
                optimizer_mu.zero_grad()
                optimizer_std.zero_grad()
                # data
                lpz, init_curve, dist2, constant = land.land_auto(loc=mu,
                                                                  scale=std,
                                                                  z_points=batch[0].to(device),
                                                                  dv=dv,
                                                                  grid=Mxy,
                                                                  model=net,
                                                                  constant=None,
                                                                  batch_size=batch_size)
                lpz.mean().backward()
                optimizer_mu.step()

                mus.append(mu.cpu().detach())
                lpzs.append(lpz.cpu())
                constants.append(constant.unsqueeze(0).cpu().detach())
                distances.append(dist2.sqrt().sum().unsqueeze(0).cpu().detach())

                if debug_mode and idx == 2:
                    break
            std_log[epoch] = std.detach().cpu()
            lpzs_log[epoch] = torch.cat(lpzs).mean().item()
            mu_log[epoch] = torch.cat(mus).mean(0)
            constant_log[epoch] = torch.cat(constants, dim=0).mean()
            distance_log[epoch] = torch.cat(distances, dim=0).mean()

            print_stderr('Epoch: {}, P(z): {:.4f}, mu: [{:.4f},{:.4f}], std: {}'.format(epoch,
                                                                                        torch.cat(lpzs).mean().item(),
                                                                                        torch.cat(mus).mean(dim=0)[0].item(),
                                                                                        torch.cat(mus).mean(dim=0)[1].item(),
                                                                                        np.round(std.data.tolist(), 4)))

        visualize.plot_training_curves(nll_log=lpzs_log,
                                       constant_log=constant_log,
                                       distance_log=distance_log,
                                       output_filename=model_folder + model_name + 'land_mu_training_curve' + str(j) + '.pdf')
        visualize.plot_mu_curve(mu_log, output_filename=model_folder + model_name + 'land_mu_plot' + str(j) + '.pdf')

    net.eval()
    for epoch in range(total_epochs + 1, total_epochs + 2*n_epochs + 1):
        total_epochs += 1
        Cs, stds, lpzs, constants, distances = [], [], [], [], []
        for idx, batch in enumerate(tqdm(train_loader)):
            optimizer_std.zero_grad()
            optimizer_mu.zero_grad()
            # data
            lpz, init_curve, dist2, constant = land.land_auto(loc=mu,
                                                              scale=std,
                                                              z_points=batch[0].to(device),
                                                              dv=dv,
                                                              constant=None,
                                                              grid=Mxy,
                                                              model=net,
                                                              batch_size=batch_size//2)
            lpz.mean().backward()
            optimizer_std.step()

            stds.append(std.cpu().detach().unsqueeze(0))
            lpzs.append(lpz.cpu())
            constants.append(constant.unsqueeze(0).cpu().detach())
            distances.append(dist2.sqrt().sum().unsqueeze(0).cpu().detach())

            if debug_mode and idx == 2:
                break

        mu_log[epoch] = mu.detach().cpu().squeeze(0)
        lpzs_log_std[epoch] = torch.cat(lpzs).mean().item()
        std_log[epoch] = torch.cat(stds).mean(0).squeeze(0)
        constant_log_std[epoch] = torch.cat(constants, dim=0).mean()
        distance_log_std[epoch] = torch.cat(distances, dim=0).mean()

        if (time.time() - start_time) > 84600:
            break

        print_stderr('Epoch: {}, P(z): {:.4f}, mu: [{:.4f},{:.4f}], std: {}'.format(epoch,
                                                                                    torch.cat(lpzs).mean().item(),
                                                                                    mu[0][0].item(),
                                                                                    mu[0][1].item(),
                                                                                    np.round(std.data.tolist(), 4)))

    visualize.plot_training_curves(nll_log=lpzs_log_std,
                                   constant_log=constant_log_std,
                                   distance_log=distance_log_std,
                                   output_filename=model_folder + model_name + 'land_std_training_curve' + str(j) + '.pdf')
    if std.dim() == 1:
        visualize.plot_std(std_log, output_filename=model_folder + model_name + 'land_std_plot' + str(j) + '.pdf')
    else:
        visualize.plot_covariance(std_log, output_filename=model_folder + model_name + 'land_cov_plot' + str(j) + '.pdf')
        visualize.plot_eigenvalues(std_log, output_filename=model_folder + model_name + 'eigenvalues_plot.pdf')

mu_save = mu.cpu().detach().numpy()
std_save = std.cpu().detach().numpy()
np.save(model_folder + 'LAND_mu_' + model_name + '.npy', mu_save, allow_pickle=True)
np.save(model_folder + 'LAND_std_' + model_name + '.npy', std_save, allow_pickle=True)