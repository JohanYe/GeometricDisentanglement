import torch
from experiment_setup import load_checkpoint, custom_dataset, make_dir
import visualize
import numpy as np
import land
import seaborn as sns
from tqdm import tqdm
import time
import experiment_setup
import utils
from geoml import stats
from os.path import join as join_path
import re
import data

sns.set_style("darkgrid")
experiment_parameters = {
    "model_dir": "bodies64-4",
    "dataset": "bodies",
    "exp_name": "init_1_sampled",
    "sampled": True,
    "load_land": False,
    "debug_mode": False,
    "hpc": False,
    "mu_init_eval": True,
}

# Experiment setup
experiment_parameters = experiment_setup.parse_args(experiment_parameters)
print(experiment_parameters)
model_dir = join_path("./model", experiment_parameters["model_dir"])
save_dir = join_path(model_dir, experiment_parameters["exp_name"])
print(save_dir)
make_dir(save_dir)  # will only create new if it doesn't exist
start_time = time.time()
torch.manual_seed(0)

# Hyperparams
batch_size = 512 if experiment_parameters["hpc"] else 64
layers = torch.linspace(28 ** 2, 2, 3).int()
num_components = 50
label_thresh = 4  # include only a subset of MNIST classes
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

## Data
if "outlier" in experiment_parameters["dataset"] or "augment" in experiment_parameters["dataset"]:
    data_path = join_path("./data", experiment_parameters["dataset"])
    x_train, y_train, N = data.load_mnist_augment(data_path)
    print("data samples loaded", N)
elif "bodies" in experiment_parameters["dataset"]:
    data_path = join_path("./data", "bodies/")
    x_train, y_train, N = data.load_bodies(data_path)
else:
    exp_regex = (re.findall(r"([a-zA-Z ]*)(\d*)", experiment_parameters["dataset"]))
    one_digit = True if len(exp_regex[1]) == 1 else False
    x_train, y_train, N = data.load_mnist_train(root="./data", label_threshold=exp_regex[1], one_digit=one_digit)

# load model
net = experiment_setup.load_model(model_dir, x_train)

with torch.no_grad():
    z = torch.chunk(net.encoder(x_train.to(device)), chunks=2, dim=-1)[0].cpu()  # [0] = mus
    z_data = torch.utils.data.TensorDataset(z)
    N_train = int(0.9 * len(z_data))
    N_test = len(z_data) - int(0.9 * len(z_data))
    train_set, test_set = torch.utils.data.random_split(z_data, [N_train, N_test])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

if experiment_parameters["load_land"]:
    mu = torch.Tensor(np.load(join_path(model_dir, 'land_mu.npy'))).to(device).requires_grad_(True)
    A = torch.Tensor(np.load(join_path(model_dir, 'land_std.npy'))).to(device).requires_grad_(True)
else:  # manual init
    mu = stats.sturm_mean(net, z.to(device), num_steps=5).unsqueeze(0)
    A = torch.Tensor(np.random.uniform(-1, 1, size=(2, 2)) / 100).to(device).float().requires_grad_(True)

# meshgrid creating
meshsize = 100 if experiment_parameters["sampled"] else 20
Mxy, dv = utils.create_grid(z, meshsize)
# curves = {}  # for init curves, but they seem useless for now

if experiment_parameters["sampled"]:
    with torch.no_grad():
        grid_prob, grid_metric, grid_metric_sum, grid_save = land.LAND_grid_prob(grid=Mxy,
                                                                                 model=net,
                                                                                 batch_size=1,
                                                                                 device=device)
        # really hacky solution
        n_faulty_grid_spots = grid_save[grid_save < 0].shape[0]
        if 0 < n_faulty_grid_spots < 5:
            grid_save[grid_save < 0] = 0
            grid_metric = grid_save.sqrt()
            grid_prob = grid_metric / grid_metric.sum()
            grid_metric_sum = grid_metric.sum()
else:
    grid_metric_sum = None


if not experiment_parameters["load_land"]:
    mus, average_loglik, stds = [], [], []
    for i in range(1):
        if experiment_parameters["mu_init_eval"]:
            mu = stats.sturm_mean(net, z.to(device), num_steps=20).unsqueeze(0)
            # mu_np = np.expand_dims(np.random.uniform(-2, 2, size=(2)), axis=0)
            # mu = torch.tensor(mu_np).to(device).float().requires_grad_(True)
        A = torch.Tensor(np.random.uniform(-1, 1, size=(2, 2)) / 100).to(device).float().requires_grad_(True)
        lpzs = []
        try:
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(train_loader)):
                    if experiment_parameters["sampled"]:
                        idxs = torch.multinomial(grid_prob, num_samples=batch_size, replacement=False)
                        Mxy_batch = Mxy[idxs].to(device)
                    else:
                        Mxy_batch = None

                    # data
                    lpz, init_curve, dist2, constant = land.land_auto(loc=mu,
                                                                      A=A,
                                                                      z_points=batch[0].to(device),
                                                                      dv=dv,
                                                                      grid=Mxy,
                                                                      model=net,
                                                                      constant=None,
                                                                      batch_size=batch_size,
                                                                      metric_grid_sum=grid_metric_sum,
                                                                      grid_sampled=Mxy_batch)
                    lpzs.append(lpz.cpu().mean())

                    if idx == 5:
                        break
            mus.append(mu.clone().detach().cpu())
            stds.append(A.clone().detach().cpu())
            average_loglik.append(np.mean(lpzs))
        except Exception as e:
            print("init seed failed")
            print(average_loglik)
            print(e)
if experiment_parameters["mu_init_eval"]:
    mu = mus[np.argmin(average_loglik)].to(device).detach().clone().requires_grad_(True)
A = stds[np.argmin(average_loglik)].to(device).clone().detach().requires_grad_(True)
print(average_loglik)
print(mus)
print(stds)

optimizer_mu = torch.optim.Adam([mu], lr=2e-3)  # , weight_decay=1e-4)
lpzs_log, mu_log, constant_log, distance_log = {}, {}, {}, {}
optimizer_std = torch.optim.Adam([A], lr=1e-3)  # , weight_decay=1e-4)
std_log, lpz_std_log = {}, {}
test_lpz_log = {}
n_epochs = 2 if experiment_parameters["debug_mode"] else 5
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

            if experiment_parameters["sampled"]:
                idxs = torch.multinomial(grid_prob, num_samples=batch_size, replacement=False)
                Mxy_batch = Mxy[idxs].to(device)
            else:
                Mxy_batch = None

            # data
            lpz, init_curve, dist2, constant = land.land_auto(loc=mu,
                                                              A=A,
                                                              z_points=batch[0].to(device),
                                                              dv=dv,
                                                              grid=Mxy,
                                                              model=net,
                                                              constant=None,
                                                              batch_size=batch_size,
                                                              metric_grid_sum=grid_metric_sum,
                                                              grid_sampled=Mxy_batch)
            lpz.mean().backward()
            if j % 2 == 0:
                optimizer_mu.step()
            else:
                optimizer_std.step()

            mus.append(mu.cpu().detach())
            lpzs.append(lpz.cpu())
            stds.append(A.cpu().detach().unsqueeze(0))
            constants.append(constant.unsqueeze(0).cpu().detach())
            distances.append(dist2.sqrt().sum().unsqueeze(0).cpu().detach())

            if experiment_parameters["debug_mode"] and idx == 2:
                break

        std_log[epoch] = A.detach().cpu()
        lpzs_log[epoch] = torch.cat(lpzs).mean().item()
        mu_log[epoch] = torch.cat(mus).mean(0)
        constant_log[epoch] = torch.cat(constants, dim=0).mean()
        distance_log[epoch] = torch.cat(distances, dim=0).mean()

        with torch.no_grad():
            lpzs_test = []
            for idx, batch in enumerate(tqdm(test_loader)):
                if experiment_parameters["sampled"]:
                    idxs = torch.multinomial(grid_prob, num_samples=batch_size, replacement=False)
                    Mxy_batch = Mxy[idxs].to(device)
                else:
                    Mxy_batch = None

                # data
                lpz, init_curve, dist2, constant = land.land_auto(loc=mu,
                                                                  A=A,
                                                                  z_points=batch[0].to(device),
                                                                  dv=dv,
                                                                  grid=Mxy,
                                                                  model=net,
                                                                  constant=None,
                                                                  batch_size=batch_size,
                                                                  metric_grid_sum=grid_metric_sum,
                                                                  grid_sampled=Mxy_batch)
                lpzs_test.append(lpz)
        test_lpz_log[epoch] = torch.cat(lpzs_test).mean().item()
        lpz_std_log[epoch] = torch.cat(lpzs).std().item()

        utils.print_stdout('Epoch: {}, P(z) train: {:.4f}, P(z) test: {:.4f} \
        , mu: [{:.4f},{:.4f}], std: {}'.format(epoch,
                                               torch.cat(lpzs).mean().item(),
                                               torch.cat(lpzs_test).mean().item(),
                                               torch.cat(mus).mean(dim=0)[0].item(),
                                               torch.cat(mus).mean(dim=0)[1].item(),
                                               np.round(A.data.tolist(), 4)))
        if epoch > 1:
            if (test_lpz_log[epoch - 1] - 0.05 * lpz_std_log[epoch]) < test_lpz_log[epoch] < (
                    test_lpz_log[epoch - 1] + 0.05 * lpz_std_log[epoch]):
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            if early_stopping_counter >= 5:
                break

        if test_lpz_log[epoch] < best_nll:
            best_nll = test_lpz_log[epoch]
            best_mu = mu.clone().detach()
            best_std = A.clone().detach()

    visualize.plot_training_curves(nll_log=lpzs_log,
                                   test_nll_log=test_lpz_log,
                                   output_filename=save_dir + '/land_mu_training_curve.pdf')
    visualize.plot_mu_curve(mu_log, output_filename=save_dir + '/land_mu_plot.pdf')
    if A.dim() == 1:
        visualize.plot_std(std_log, output_filename=save_dir + '/land_std_plot.pdf')
    else:
        visualize.plot_covariance(std_log,
                                  output_filename=save_dir + '/land_cov_plot.pdf')
        visualize.plot_eigenvalues(std_log, output_filename=save_dir + '/eigenvalues_plot.pdf')

mu_save = best_mu.cpu().detach().numpy()
std_save = best_std.cpu().detach().numpy()
np.save(save_dir + '/LAND_mu_.npy', mu_save, allow_pickle=True)
np.save(save_dir + '/LAND_std_.npy', std_save, allow_pickle=True)
