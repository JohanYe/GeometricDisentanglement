#!/usr/bin/env python3
import torch
from torch.optim.optimizer import Optimizer, required


class RiemSGD(Optimizer):
    r"""Implements stochastic gradient descent on Riemannian manifolds.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        manifolds (iterable): iterable of objects of class 'Manifold' over which
            the parameters are to be optimized.
        lr (float): learning rate
    """

    def __init__(self, params, manifolds, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(RiemSGD, self).__init__(params, defaults)
        self.manifolds = manifolds # XXX: what's the pytorch way of doing this?

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, manifold in zip(self.param_groups, self.manifolds):
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                with torch.no_grad():
                    p.copy_(manifold.expmap(p, group['lr'] * d_p).view(p.shape))

        return loss
