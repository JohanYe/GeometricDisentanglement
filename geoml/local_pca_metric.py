#!/usr/bin/env python3
import torch
import numpy as np
from geoml.manifold import Manifold

class LocalVarMetric(Manifold):
    r"""
    A class for computing the local inverse-variance metric described in

        A Locally Adaptive Normal Distribution
        Georgios Arvanitidis, Lars Kai Hansen, and Søren Hauberg.
        Neural Information Processing Systems, 2016.

    along with associated quantities. The metric is diagonal with elements
    corresponding to the inverse (reciprocal) of the local variance, which
    is defined as

        var(x) = \sum_n w_n(x) * (x_n - x)^2 + rho
        w_n(x) = exp(-|x_n - x| / 2sigma^2)
    """

    def __init__(self, data, sigma, rho, device=None):
        """
        Class constructor.

        Mandatory inputs:
            data:   The data from which local variance will be computed.
                    This should be a NxD torch Tensor correspondong to
                    N observations of dimension D.
            sigma:  The width of the Gaussian window used to define
                    locality when computing weights. This must be a positive
                    scalar.
            rho:    The added bias in the local variance estimate. This
                    must be a positive scalar (usually a small number, e.g. 1e-4).

        Optional inputs:
            device: The torch device on which computations will be performed.
                    Default: None
        """
        super().__init__()
        self.data = data
        self.sigma2 = sigma**2
        self.rho = rho
        self.device = device

    def metric(self, c, return_deriv=False):
        """
        Evaluate the local inverse-variance metric tensor at a given set of points.

        Mandatory input:
          c:              A PxD torch Tensor containing P points of dimension D where
                          the metric will be evaluated.

        Optional input:
          return_deriv:   If True the function will return a second output containing
                          the derivative of the metric tensor. This will be returned
                          in the form of a PxDxD torch Tensor.
                          Default: False

        Output:
          M:              The diagonal elements of the inverse-variance metric
                          represented as a PxD torch Tensor.
        """
        X = self.data # NxD
        N = X.shape[0]
        P, D = c.shape
        sigma2 = self.sigma2
        rho = self.rho
        K = 1.0 / ((2.0*np.pi*sigma2)**(D/2.0))

        # Compute metric
        M = [] #torch.empty((P, D)) # metric
        dMdc = [] # derivative of metric in case it is requested
        for p in range(P):
            delta  = X - c[p] # NxD
            delta2 = (delta)**2 # NxD
            dist2 = delta2.sum(dim=1) # N
            w_p = K * torch.exp(-0.5*dist2/sigma2).reshape((1, N)) # 1xN
            S = w_p.mm(delta2) + rho # D
            m = 1.0 / S # D
            M.append(m)
            if return_deriv:
                weighted_delta = (w_p/sigma2).reshape(-1, 1).expand(-1, D) * delta # NxD
                dSdc = 2.0 * torch.diag(w_p.mm(delta).flatten()) - weighted_delta.t().mm(delta2) # DxD
                dM = dSdc.t() * (m**2).reshape(-1, 1).expand(-1, D) # DxD
                dMdc.append(dM.reshape(1, D, D))

        if return_deriv:
            return torch.cat(M), torch.cat(dMdc, dim=0)
        else:
            return torch.cat(M)

    def curve_energy(self, c):
        """
        Evaluate the energy of a curve represented as a discrete set of points.

        Input:
            c:      A discrete set of points along a curve. This is represented
                    as a PxD or BxPxD torch Tensor. The points are assumed to be ordered
                    along the curve and evaluated at equidistant time points.

        Output:
            energy: The energy of the input curve.
        """
        sh = c.shape
        if len(c.shape) is 2:
            c.unsqueeze_(0) # add batch dimension if one isn't present
        energy = torch.zeros(1)
        for b in range(c.shape[0]):
            M = self.metric(c[b, :-1]) # (P-1)xD
            delta1 = (c[b, 1:] - c[b, :-1])**2 # (P-1)xD
            energy += (M * delta1).sum()
        return energy

    def curve_length(self, c):
        """
        Evaluate the length of a curve represented as a discrete set of points.

        Input:
            c:      A discrete set of points along a curve. This is represented
                    as a PxD torch Tensor. The points are assumed to be ordered
                    along the curve and evaluated at equidistant indices.

        Output:
            length: The length of the input curve.
        """
        M = self.metric(c[:-1]) # (P-1)xD
        delta1 = (c[1:] - c[:-1])**2 # (P-1)xD
        length = (M * delta1).sum(dim=1).sqrt().sum()
        return length

    def geodesic_system(self, c, dc):
        """
        Evaluate the 2nd order system of ordinary differential equations that
        govern geodesics.

        Inputs:
            c:      A DxN torch Tensor of D-dimensional points on the manifold.
            dc:     A DxN torch Tensor of first derivatives at the points specified
                    by the first input argument.

        Output:
            ddc:    A DxN torch Tensor of second derivatives at the specified locations.
        """
        D, N = c.shape
        M, dM = self.metric(c.t(), return_deriv=True) # [NxD, NxDxD]

        # Prepare the output
        ddc = [] #torch.zeros(D, N) # DxN

        # Evaluate the geodesic system
        for n in range(N):
            dMn = dM[n] # DxD
            ddc_n = -0.5*(2.0*(dMn * dc[:, n].reshape(-1, 1).expand(-1, D)).mv(dc[:, n])
                        - dMn.t().mv(dc[:, n]**2)) / M[n].flatten()
            ddc.append(ddc_n.reshape(D, 1))

        ddc_tensor = torch.cat(ddc, dim=1) # DxN
        return ddc_tensor
