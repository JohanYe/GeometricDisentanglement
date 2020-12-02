# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:36:14 2018

@author: nsde
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

class VAE_bernoulli(nn.Module):
    def __init__(self, latent_size, image_dim=[1, 28, 28], device="cpu", learning_rate=1e-4):
        super(VAE_bernoulli, self).__init__()
        self.image_dim = image_dim
        self.latent_size = latent_size
        self.enc_mu = nn.Sequential(nn.Linear(784, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, latent_size))

        self.enc_var = nn.Sequential(nn.Linear(784, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, latent_size),
                                     nn.Softplus())

        self.dec_mu = nn.Sequential(nn.Linear(latent_size, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 784),
                                    nn.Sigmoid())

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)

        if device == "gpu":
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z_mu = self.enc_mu(x)
        z_var = self.enc_var(x)
        z = z_mu + torch.randn_like(z_var) * z_var.sqrt()
        x_mu = self.dec_mu(z)
        return x_mu, z_mu, z_var

    def loss_function(self, x, x_mu, z_mu, z_var):
        recon_error = nn.functional.binary_cross_entropy(x_mu, x.view(x.shape[0], -1), reduction='sum')
        KL = -0.5 * torch.sum(1 + z_var.log() - z_mu.pow(2) - z_var)
        loss = recon_error + KL
        return loss, recon_error, KL

    def train(self, X, n_epochs=100,  batch_size=64, verbose=True, labels=None):
        n_batch = int(np.ceil(X.shape[0] / batch_size))
        for e in range(1, n_epochs+1):
            loss_avg = 0
            x_enc = [ ]
            for idx in range(n_batch):
                self.optimizer.zero_grad()
                x = X[idx*batch_size:(idx+1)*batch_size]
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
                x_mu, z_mu, z_var = self.forward(x)
                loss, recon, kl = self.loss_function(x, x_mu, z_mu, z_var)
                loss.backward()
                self.optimizer.step()
                loss_avg += loss.item()

                if verbose and self.latent_size == 2:
                    x_enc.append(z_mu)

            if verbose:
                print("Epoch {0}/{1}, Loss {2}".format(e, n_epochs, loss_avg / X.shape[0]))
                save_image(torch.cat((x[:10].view((10, *self.image_dim)), x_mu[:10].view(10, *self.image_dim)),
                           dim=0).cpu(), '../images/recon_'+str(e) + '.png', nrow=10)

                if self.latent_size == 2:
                    # Save grid of images
                    z = torch.stack([array.flatten() for array in torch.meshgrid(
                        [torch.linspace(-3, 3, 40), torch.linspace(-3, 3, 40)])]).t()
                    x_mu = self.dec_mu(z.to(self.device))
                    save_image(x_mu.view(1600, *self.image_dim).cpu(), '../images/grid_'+str(e)+'.png', nrow=40)
                    # Save point cloud
                    x_enc = torch.cat(x_enc, dim=0).detach().cpu().numpy()
                    fig=plt.figure(frameon=False)
                    plt.scatter(x_enc[:,0], x_enc[:,1], c=labels)
                    if labels is not None: plt.colorbar()
                    plt.savefig('../images/latent_'+str(e)+'.png')
                    plt.close(fig)

    def save(self, filename):
        #save_model(self, self.optimizer, filename)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
             }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def intrinsic_coordinates(self, points): # encoder
        points = points.to(self.device)
        z_mu = self.enc_mu(points)
        z_var = self.enc_var(points)
        coors = torch.cat((z_mu, z_var), dim=1)
        return coors.cpu()

    def embed(self, coords): # decoder
        coords = coords.to(self.device)
        points_mu = self.dec_mu(coords)
        return points_mu.cpu()

    def curve_energy(self, coords):
        # coords: Nx(D-1)
        points = self.embed(coords) # NxD
        energy = (points[1:] - points[:-1]).pow(2).sum()
        return energy
