import torch
import torchvision.datasets
import model
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image
from torchvision import transforms as transforms
from os.path import join as join_path
import data
import utils
import experiment_setup

sns.set_style("darkgrid")
experiment_parameters = {
    "model_dir": "unnamed_model",
    "dataset": "bodies",
    "hidden_layer": 0,
    "num_components": 4500,
    "std_epochs":2000,
    "beta_constant":0.5,
    "inv_maxstd":0.5,
}

# Experiment setup
# experiment_parameters = experiment_setup.parse_args_vae(experiment_parameters)
model_dir = join_path("./model", experiment_parameters["model_dir"])
experiment_setup.make_dir(model_dir)

# HyperParams
batch_size = 512
layers = utils.get_layer_sizes(experiment_parameters)
label_thresh = 1 # include only a subset of MNIST classes
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

## Data
x_train, y_train, N, x_test, y_test, N_test = data.load_data(experiment_parameters, root="./data")
train_loader, test_loader = data.data_split(x_train, x_test,batch_size)

# Fit mean network
if experiment_parameters["dataset"] == "bodies":
    model = model.VAE_bodies(x_train, layers, num_components=experiment_parameters["num_components"], device=device)
else:
    model = model.VAE(x_train, layers, num_components=experiment_parameters["num_components"], device=device)
model.fit_mean(train_loader, num_epochs=5, num_cycles=1, max_kl=1)

# fit std
model.init_std(x_train,
               inv_maxstd=experiment_parameters["inv_maxstd"],
               beta_constant=torch.Tensor([experiment_parameters["beta_constant"]]),
               component_overwrite=experiment_parameters["num_components"])
model.fit_std(train_loader, num_epochs=experiment_parameters["std_epochs"])


save_checkpoint({'state_dict': model.state_dict(), 'gmm_means': model.gmm_means, 'gmm_cv': model.gmm_covariances,
                'weights': model.clf_weights, 'num_components': num_components, "layers":layers,
                'inv_maxstd':inv_maxstd, 'beta_constant':beta_constant, "beta_override":beta}, './model/bodies64-4')