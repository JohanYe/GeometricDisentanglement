import torch
import torchvision.datasets
import model
from utils import load_checkpoint, custom_dataset, print_stdout, make_dir
import visualize
import numpy as np
import land
import seaborn as sns
from tqdm import tqdm
import time

sns.set_style("darkgrid")
model_folder = "./model/mnist2-3/"
model_name = 'init_3_sampled'
print(model_folder + model_name)
make_dir(model_folder + model_name)
save_dir = model_folder + model_name + "/"
sampled = True
load_land = False
hpc = True
fast_train = False
debug_mode = False
full_cov = True
start_time = time.time()
torch.manual_seed(0)

batch_size = 512 if hpc else 64
layers = torch.linspace(28 ** 2, 2, 3).int()
num_components = 50
label_thresh = 4  # include only a subset of MNIST classes
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

## Data
mnist_train = torchvision.datasets.MNIST('data/', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float() / 255
y_train = mnist_train.targets
if "0" in model_folder:
    label_thresh = 2  # include only a subset of MNIST classes
    idx = y_train < label_thresh  # only use digits 0, 1, 2, ...
    print("samples:", idx.sum())
elif "vae" in model_folder:
    label_thresh = 4  # include only a subset of MNIST classes
    idx = y_train < label_thresh  # only use digits 0, 1, 2, ...
    print("samples:", idx.sum())
elif "mnist2" in model_folder:
    label_thresh = 2  # include only a subset of MNIST classes
    idx = y_train == label_thresh  # only use digits 0, 1, 2, ...
    print("samples:", idx.sum())
else:
    label_thresh = 1  # include only a subset of MNIST classes
    idx = y_train == label_thresh  # only use digits 0, 1, 2, ...
    print("samples:", idx.sum())
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
    N_train = int(0.9 * len(z_data))
    N_test = len(z_data) - int(0.9 * len(z_data))
    train_set, test_set = torch.utils.data.random_split(z_data, [N_train, N_test])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

if load_land:
    mu = torch.Tensor(np.load(model_folder + 'land_mu.npy')).to(device).requires_grad_(True)
    std = torch.Tensor(np.load(model_folder + 'land_std.npy')).to(device).requires_grad_(True)
else:  # manual init
    mu_np = np.expand_dims(np.array([0, 0]), axis=0)
    mu = torch.tensor(mu_np).to(device).float().requires_grad_(True)
    if full_cov:
        std = torch.Tensor(np.random.uniform(-1, 1, size=(2, 2)) / 100).to(device).float().requires_grad_(True)
    else:
        std = torch.tensor([40.]).to(device).float().requires_grad_(True)

# meshgrid creating
meshsize = 100 if sampled else 20
ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize)
ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize)
Mx, My = torch.meshgrid(ran0, ran1)
Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
Mxy.requires_grad = False
dv = (ran0[-1] - ran0[0]) * (ran1[-1] - ran1[0]) / (meshsize ** 2)
curves = {}

if sampled:
    with torch.no_grad():
        grid_prob, grid_metric, grid_metric_sum = land.LAND_grid_prob(grid=Mxy,
                                                                      model=net,
                                                                      batch_size=1024,
                                                                      device=device)
        grid = Mxy.clone()
else:
    grid_metric_sum = None

optimizer_mu = torch.optim.Adam([mu], lr=1e-3)  # , weight_decay=1e-4)
lpzs_log, mu_log, constant_log, distance_log = {}, {}, {}, {}
optimizer_std = torch.optim.Adam([std], lr=5e-4)  # , weight_decay=1e-4)
std_log, lpz_std_log = {}, {}
test_lpz_log = {}
n_epochs = 2 if debug_mode else 5
total_epochs = 0
early_stopping_counter = 0
best_nll = np.inf

net.eval()
net = net
for j in range(40):
    for epoch in range(total_epochs + 1, total_epochs + n_epochs + 1):
        total_epochs += 1
        Cs, mus, stds, lpzs, constants, distances = [], [], [], [], [], []
        for idx, batch in enumerate(tqdm(train_loader)):
            optimizer_mu.zero_grad()
            optimizer_std.zero_grad()

            if sampled:
                idxs = torch.multinomial(grid_prob, num_samples=batch_size, replacement=False)
                Mxy_batch = Mxy[idxs].to(device)
            else:
                Mxy_batch = None

            # data
            lpz, init_curve, dist2, constant = land.land_auto(loc=mu,
                                                              scale=std,
                                                              z_points=batch[0].to(device),
                                                              dv=dv,
                                                              grid=Mxy,
                                                              model=net,
                                                              constant=None,
                                                              batch_size=batch_size,
                                                              grid_sum=grid_metric_sum,
                                                              grid_batch=Mxy_batch)
            lpz.mean().backward()
            if j % 2 == 0:
                optimizer_mu.step()
            else:
                optimizer_std.step()

            mus.append(mu.cpu().detach())
            lpzs.append(lpz.cpu())
            stds.append(std.cpu().detach().unsqueeze(0))
            constants.append(constant.unsqueeze(0).cpu().detach())
            distances.append(dist2.sqrt().sum().unsqueeze(0).cpu().detach())

            if debug_mode and idx == 2:
                break

        std_log[epoch] = std.detach().cpu()
        lpzs_log[epoch] = torch.cat(lpzs).mean().item()
        mu_log[epoch] = torch.cat(mus).mean(0)
        constant_log[epoch] = torch.cat(constants, dim=0).mean()
        distance_log[epoch] = torch.cat(distances, dim=0).mean()

        with torch.no_grad():
            lpzs_test = []
            for idx, batch in enumerate(tqdm(test_loader)):
                if sampled:
                    idxs = torch.multinomial(grid_prob, num_samples=batch_size, replacement=False)
                    Mxy_batch = Mxy[idxs].to(device)
                else:
                    Mxy_batch = None

                # data
                lpz, init_curve, dist2, constant = land.land_auto(loc=mu,
                                                                  scale=std,
                                                                  z_points=batch[0].to(device),
                                                                  dv=dv,
                                                                  grid=Mxy,
                                                                  model=net,
                                                                  constant=None,
                                                                  batch_size=batch_size,
                                                                  grid_sum=grid_metric_sum,
                                                                  grid_batch=Mxy_batch)
                lpzs_test.append(lpz)
        test_lpz_log[epoch] = torch.cat(lpzs_test).mean().item()
        lpz_std_log[epoch] = torch.cat(lpzs).std().item()

        print_stdout('Epoch: {}, P(z) train: {:.4f}, P(z) test: {:.4f} \
        , mu: [{:.4f},{:.4f}], std: {}'.format(epoch,
                                               torch.cat(lpzs).mean().item(),
                                               torch.cat(lpzs_test).mean().item(),

                                               torch.cat(mus).mean(dim=0)[0].item(),
                                               torch.cat(mus).mean(dim=0)[1].item(),
                                               np.round(std.data.tolist(), 4)))
        if epoch > 1:
            if (test_lpz_log[epoch - 1] - 0.2 * lpz_std_log[epoch]) < test_lpz_log[epoch] < (
                    test_lpz_log[epoch - 1] + 0.2 * lpz_std_log[epoch]):
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            if early_stopping_counter >= 5:
                break

        if test_lpz_log[epoch] < best_nll:
            best_nll = test_lpz_log[epoch]
            best_mu = mu.clone().detach()
            best_std = std.clone().detach()

    visualize.plot_training_curves(nll_log=lpzs_log,
                                   constant_log=constant_log,
                                   distance_log=distance_log,
                                   output_filename=save_dir + 'land_mu_training_curve.pdf')
    visualize.plot_mu_curve(mu_log, output_filename=save_dir + 'land_mu_plot.pdf')
    if std.dim() == 1:
        visualize.plot_std(std_log, output_filename=save_dir + 'land_std_plot.pdf')
    else:
        visualize.plot_covariance(std_log,
                                  output_filename=save_dir + 'land_cov_plot.pdf')
        visualize.plot_eigenvalues(std_log, output_filename=save_dir + 'eigenvalues_plot.pdf')

mu_save = best_mu.cpu().detach().numpy()
std_save = best_std.cpu().detach().numpy()
np.save(save_dir + 'LAND_mu_.npy', mu_save, allow_pickle=True)
np.save(save_dir + 'LAND_std_.npy', std_save, allow_pickle=True)