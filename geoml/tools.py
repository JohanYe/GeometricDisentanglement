#!/usr/bin/env python3
import torch
from numpy import pi

class LocationScaledTdistribution:
    def __init__(self, mu, alpha, beta):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.log_const = alpha * torch.log(beta) + \
                         torch.lgamma(alpha + 0.5) - \
                         torch.lgamma(alpha) - \
                         0.5*torch.log(torch.tensor(2.0*pi))

    def log_prob(self, x):
        return self.log_const - (self.alpha + 0.5) * torch.log(self.beta + 0.5*(x - self.mu).pow(2))
