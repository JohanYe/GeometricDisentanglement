#!/usr/bin/env python3
import torch
from .curve import *
from .optim import RiemSGD
from .manifold import Manifold
from itertools import repeat
from torch.distributions.kl import register_kl
from torch.distributions.multivariate_normal import MultivariateNormal
from .functions import Hyp1f1
from scipy.special import gamma
import numpy as np
from tqdm import tqdm

def sturm_mean(manifold, data, num_steps=None):
    """
    Compute the intrinsic mean of a data set using Sturm's recursive algorithm.

    Mandatory inputs:
    manifold:   a Manifold object representing the space in which the mean is
                to be computed.
    data:       a N-by-D matrix of N observations in D dimensions.

    Optional inputs:
    num_steps=N:    number of steps taken by the algorithm. By default
                    one pass is taken through the entire data set.
    """
    device = data.device
    if num_steps is None:
        num_steps = data.shape[0]
    N, D = data.shape
    idx = torch.randint(high=N, size=(num_steps,), dtype=torch.long)
    mu = data[idx[0]]
    n = 1.0
    for m in tqdm(range(1, num_steps)):
        c, success = manifold.connecting_geodesic(mu.view(1, -1), data[idx[m]].view(1, -1))
        if success:
            n += 1.0
            alpha = torch.tensor(n).reciprocal().to(device)
            mu = c(alpha).detach()
    return mu.flatten()

def intrinsic_mean(manifold, data, lr, init_mu=None, num_steps=10, batch_size=None):
    """
    Compute the intrinsic mean of a data set using Riemannian stochastic gradient descent.

    Mandatory inputs:
    manifold:   a Manifold object representing the space in which the mean is
                to be computed.
    data:       a N-by-D matrix of N observations in D dimensions.

    Optional inputs:
    num_steps=N:    number of steps taken by the algorithm. By default
                    one pass is taken through the entire data set.
    """
    if init_mu is None:
        mu = data.mean(dim=0).view(1, -1)
    else:
        mu = init_mu.clone().detach()
    mu.requires_grad_(True)

    if batch_size is None:
        batch_size = data.shape[0]

    opt = RiemSGD([mu], [manifold], lr=lr)
    for iteration in range(num_steps):
        batch_idx = torch.randint(high=data.shape[0], size=[batch_size], dtype=torch.long)
        loss = torch.zeros(1)
        for b in batch_idx:
            loss += manifold.dist2(mu, data[b].view(1, -1))
        loss /= batch_size
        loss.backward()

        print(iteration, loss.detach(), mu.grad.norm())

        opt.step()
    return mu

# SGD of median (same as mean but apply a sqrt)

# PGA

# ProbPGA? or a variant thereof

# k-Means
def intrinsic_kmeans(manifold, data, K, lr, init_mus=None, num_steps=10, batch_size=None):
    if init_mus is None:
        idx = torch.randint(high=data.shape[0], size=[K], dtype=torch.long)
        mus = [data[i].clone().requires_grad_(True) for i in idx]
    else:
        mus = init_mus

    if batch_size is None:
        batch_size = data.shape[0]

    manifolds = repeat(manifold, K)
    opt = RiemSGD(mus, manifolds, lr=lr)
    for iteration in range(num_steps):
        batch_idx = torch.randint(high=data.shape[0], size=[batch_size], dtype=torch.long)
        loss = torch.zeros(1)
        for b in batch_idx:
            distances = torch.zeros(K)
            for k in range(K):
                distances[k] = manifold.dist2(mus[k].view(1, -1), data[b].view(1, -1))
            loss += distances.min()
        loss /= batch_size
        loss.backward()

        print(iteration, loss.detach(), [mu.grad.norm() for mu in mus])

        opt.step()
    return mus

# Principal curves?

# a LAND class
#   -- fit
#   -- evaluate log_likelihood
#   -- sample

# a Brownian motion class??
class IsotropicBrownianMotion:
    def __init__(self, manifold, mean, variance=1.0):
        self.manifold = manifold

        if isinstance(mean, torch.Tensor):
            self.mean = mean.view(1, -1) # 1x(dim)
        else:
            self.mean = torch.tensor(mean).view(1, -1) # 1x(dim)
        self.dim = self.mean.numel()
        self.log_volume_at_mean = self.manifold.log_volume(self.mean) # scalar

        if isinstance(variance, torch.Tensor):
            self.variance = variance
        else:
            self.variance = torch.tensor(variance)

    def sample(self, num_samples=1, num_steps=10, retain_path=False):
        """
        Return a sample from an isotropic Brownian motion.

        Input:
            num_samples:    a positive integer indicating number of samples.
                            Default is 1.
            num_steps:      Number of discrete steps taken by the Brownian
                            motion. Larger values give a more precise simulation
                            at the cost of increased computations.
                            Default is 10.
            retain_path:    If True, the entire random walk path is returned,
                            otherwise only the end-position is returned.
                            Default is False.
        Output:
            S:              A (num_samples)x(dim) torch Tensor containing the
                            requested sample (if retain_path=False). If retain_path
                            is True, then a (num_samples)x(num_steps)x(dim) Tensor
                            is instead returned.
        """
        S = self.mean.clone().repeat(num_samples, 1) # (num_samples)x(dim)
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            M = self.manifold.metric(S) # (num_samples)x(dim)x(dim)
            C = self.variance * dt * M.inverse() # (num_samples)x(dim)x(dim)
            S = MultivariateNormal(loc=S, covariance_matrix=C).rsample() # (num_samples)x(dim)
        return S

    def log_prob(self, x):
        """
        Compute the log-likelihood under isotropic Brownian motion.

        Input:
            x:    A Nx(dim) torch Tensor.

        Output:
            L:    A (N) torch Tensor with the log-likelihood of the data.
        """
        from numpy import pi
        scaling = -0.5*self.dim * (2.0*pi*self.variance).log() # scalar
        logH0 = 0.5*self.manifold.log_volume(x) - 0.5*self.log_volume_at_mean # (N)
        dist = -0.5*self.manifold.dist2(x, self.mean) / self.variance
        return scaling + logH0 + dist

    def fit(self, x, num_epochs=1000):
        """
        Perform maximum likelihood estimation of distribution parameters that
        store gradients.

        Input:
            x:              A Nx(dim) torch Tensor.
            num_epochs:     Number of iterations applied by the optimizer.

        Output:
            loss:           The log-likelihood of the data in the last pass
                            of the optimization loop.

        Algorithmic note:
            This method optimize the log-likelihood of data using a gradient-based
            optimization. The method will only change distribution parameters
            that have requires_grad=True.
        """
        optimizer = torch.optim.Adam([self.mean, self.variance], lr=1e-3)
        N = x.shape[0]
        for _ in range(num_epochs):
            sum_loss = 0.0
            for n in range(N):
                data = x[n]
                def closure():
                    optimizer.zero_grad()
                    loss = -self.log_prob(data)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure=closure)
                sum_loss += loss.item()
        return -sum_loss

@register_kl(IsotropicBrownianMotion, IsotropicBrownianMotion)
def kl_IBM_IBM(p, q, num_samples=1, num_steps=1):
    z = p.sample(num_samples=num_samples, num_steps=num_steps) # (num_samples)x(dim)
    lpz = p.log_prob(z) # (num_samples)
    lqz = q.log_prob(z) # (num_samples)
    KL = (lpz - lqz).mean() # scalar
    return KL

# Mixture Models (semi-supservised; missing data)


class NonCentralNakagami:
    """ Non central Nakagami distribution computed for data points z.
    inputs:
        - var: the variance of z (vector of size: N)
        - mu: the mean of z (matrix of size: NxD)
    source:
        S. Hauberg, 2018, "The non-central Nakagami distribution"
    """
    def __init__(self, mu, var):
        self.D = mu.shape[1]
        self.var = var
        self.omega = (mu**2).sum(1)/var

    def expectation(self):
        # eq 2.9. expectation = E[|z|], when z ~ N(mu, var)
        var, D, omega = self.var, self.D, self.omega
        const = np.sqrt(2)
        term_gamma = gamma((D+1)/2)/gamma(D/2)
        term_hyp1f1 = Hyp1f1.apply(torch.tensor(-1/2), torch.tensor(D/2), -1/2*omega)
        expectation = torch.sqrt(var) * const * term_gamma * term_hyp1f1
        return expectation

    def variance(self):
        # eq 2.11. variance = var[|z|], when z ~ N(mu, var)
        var, D, omega = self.var, self.D, self.omega
        term_gamma = gamma((D+1)/2)/gamma(D/2)
        term_hyp1f1 = Hyp1f1.apply(torch.tensor(-1/2), torch.tensor(D/2), -1/2*omega)
        variance = var * (omega + D - 2*(term_gamma*term_hyp1f1)**2)
        return variance
