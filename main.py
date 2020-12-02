import torch
import torchvision.datasets
import model
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_style("darkgrid")

n_epochs = 250
batch_size = 64
label_thresh = 4
num_cycles = 1
switch_epoch = n_epochs // 3

# model stuff
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.BasicVAE().to(device)

# data stuff
mnist_train = torchvision.datasets.MNIST('data/', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float() / 255
y_train = mnist_train.targets
idx = y_train < label_thresh  # only use digits 0, 1, 2, ...
num_classes = y_train[idx].unique().numel()
x_train = x_train[idx]
y_train = y_train[idx]
N = x_train.shape[0]
train_data = torch.utils.data.TensorDataset(x_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

train_loss_mean = net.fit_mean(train_loader, num_epochs=n_epochs, num_cycles=1)
net.init_std_naive()
train_loss_std = net.fit_std(train_loader, num_epochs=n_epochs)

# plotting
beta = []
for epoch in range(n_epochs):
    beta.append(min(1.0, max(0.0, 2.0 * (epoch - switch_epoch) / (n_epochs - switch_epoch))))

# recons
fig, ax = plt.subplots(1,3,figsize=(10,7))
x = np.arange(1, n_epochs+1)
ax[0].plot(x, train_loss_mean)
ax[0].set_title('Training plots')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('-ELBO')

x = np.arange(1, n_epochs+1)
ax[].plot(x, train_loss_std)
ax[1].set_title('Training plots')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Negative log likelihood')

ax[2].plot(x, beta)
ax[2].set_xlabel('Epoch')
ax[2].set_ylabel('Warm up coefficient')
ax[2].set_title('Beta Annealing')
plt.show()



from torch.distributions.kl import kl_divergence as KL
q_zGx = net.encode(data)
z = q_zGx.rsample()
p_xGz = net.decode(z)
ELBO = torch.mean(p_xGz.log_prob(data) - KL(q_zGx, net.prior), dim=0)
print(p_xGz.log_prob(data), KL(q_zGx, net.prior))

x_plot = p_xGz.mean.cpu().detach().numpy().reshape(63,28,28)
n_rows = 5
fig, axs = plt.subplots(n_rows, n_rows, figsize=(7, 7), constrained_layout=True)
for i in range(n_rows):
    for j in range(n_rows):
        axs[i, j].imshow(x_plot[n_rows * i + j], cmap='gray')
        axs[i, j].axis('off')
        axs[i, j].axis('off')