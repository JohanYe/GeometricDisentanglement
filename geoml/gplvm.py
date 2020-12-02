# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10, 2018

@author: cife
"""

import numpy as np
import torch
import pyro.contrib.gp as gp
import pyro

import pickle
from sklearn.metrics import pairwise_distances
from geoml import *
from torch.autograd import Variable
from numpy.linalg import eigvalsh
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.nn import functional


class gplvm(Manifold):
    """Class that takes a gplvm model as input and allows for computation of curve energy"""

    def __init__(self, data,object, device=None, mode='riemannian'):#,modeltype='gplvm'):
        self.model = object     #
        self.device = device
        self.mode = mode        # Riemannian or Finslerian
        self.data = data


    def update_modelparms(self,modeltype):
        mp = dict()

        if modeltype == 'gplvm':
            mp['Y'] = self.model.base_model.__dict__['y'].clone().detach().t().double()

            mp['noise_unconstrained']      = self.model.base_model.noise_unconstrained.double()
            # TODO: Delete kernel.
            mp['kernel.variance_unconstrained'] = self.model.base_model.kernel.variance_unconstrained.double()
            mp['kernel.lengthscale_unconstrained'] = self.model.base_model.kernel.lengthscale_unconstrained.double()

        elif modeltype == 'bayesian':
           mp['Y']                      = self.model.__dict__['y'].double()
           mp['Xu']                     = self.model.Xu.double()
           #mp['X_loc']                  = self.model.X_loc.double()
           mp['X_scale_unconstrained']  = self.model.X_scale_unconstrained.double()
           mp['noise_unconstrained']      = self.model.noise_unconstrained.double()
           # TODO: Delete kernel.
           mp['kernel.variance_unconstrained'] = self.model.kernel.variance_unconstrained.double()
           mp['kernel.lengthscale_unconstrained'] = self.model.kernel.lengthscale_unconstrained.double()


        mp['X_loc']                  = self.model.X_loc.double()
        mp['N'],mp['D'] = mp['Y'].shape
        print(mp.keys())
        return mp

    def __str__(self):
        str = 'Customised print method'
        # TODO: Write the string method.
        # Hyperparameters
        # Dimensionality
        # Convergence
        #print("y shape (N x D):",y.shape)
        #str = str(self.obj.__str__) + '\n\n' + str(self.obj.optimization_runs[0].status)
        return str

    def pairwise_distances(self,x, y=None):
        """
        OBSOLETE
        Helper function.
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """

        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def evaluateKernel(self,xstar,x):
        """
        OBSOLETE
        Helper function.k
        For the RBF kernel, evaluate the kernel using the pairwise distances
        Returns the evaluated kernel of the latent variables and the new point
        """
        # BUG: Wrong calculation
        # QUESTION: I kind of feel like there should be a smarter way of doing this.
        # QUESTION: I think there should be a squareroot somewhere!
        r = self.pairwise_distances(xstar,x)
        var = self.model.kernel.variance_unconstrained.exp()
        l = self.model.kernel.lengthscale_unconstrained.exp()

        # HACK: Does not work for ard!!!
        # FIX: Does not work for ard!!! Fix that it cannot handle non-scalar lengthscales
        ksx = var * torch.exp((-0.5*r/l**2))
        return ksx

    def evaluateDiffKernel_batch(self, xstar, X, kernel_type='rbf'):
        """
        Compute the differentiation of the kernel at a single point

        inputs:
            - xstar: data to be predicted (size: nof_points x d)
            - X: training data (size: Nxd)
        outputs:
            - dK: first derivative of the kernel (matrix, size: dxN)
            - ddK: second derivative of the kernel (matrix, size: dxd)

        Note: Converting to double means that the output is equal to the non-betached function down to 1e-7 rather than 1e-5.
        """
        N_train, d = X.shape
        if kernel_type=='rbf': # faster for RBF
            if len(xstar.shape) <2:
                xstar = xstar.unsqueeze(0).double()
            nof_points =  xstar.shape[0]
            xstar = xstar.double()

            var = self.model.kernel.variance_unconstrained.exp().double()
            l = self.model.kernel.lengthscale_unconstrained.exp().double()
            #ksx = self.evaluateKernel(xstar,X.double()).double()
            ksx = self.model.kernel.forward(xstar,X.double())


            X = torch.cat(nof_points*[X.unsqueeze(0)]).double()
            xstar = torch.cat(N_train*[xstar.unsqueeze(0)]).transpose(0,1)

            hej = xstar-X
            hest = torch.cat(d*[ksx.unsqueeze(0)]).transpose(0,1).transpose(1,2)

            dK = -l**-2*hej*hest
            ddK = (l**-2*var)*torch.eye(d).to(torch.float64)

        else:
            # TODO: @Alison: I think there is a mistake in evaluateKernel, let's use forward! I guess we could just parse that instead?
            kernel_jac = functional.jacobian(self.evaluateKernel, (xstar, X), create_graph=True)
            kernel_hes = functional.hessian(self.evaluateKernel, (xstar, xstar), create_graph=True)
            dK = torch.reshape(torch.squeeze(kernel_jac[0]), (N_train, d)).T
            ddK = torch.reshape(torch.squeeze(kernel_hes[0][1]), (d, d))
        return dK, ddK

    def evaluateDiffKernel(self, xstar, X, kernel_type='rbf'):
        """
        OBSOLETE
        Compute the differentiation of the kernel at a single point

        inputs:
            - xstar: data to be predicted (size: 1xd)
            - X: training data (size: Nxd)
        outputs:
            - dK: first derivative of the kernel (matrix, size: dxN)
            - ddK: second derivative of the kernel (matrix, size: dxd)
        """
        N_train, d = X.shape
        if kernel_type=='rbf': # faster for RBF

            var = self.model.base_model.kernel.variance_unconstrained.exp()
            l = self.model.base_model.kernel.lengthscale_unconstrained.exp()
            #ksx = self.evaluateKernel(xstar,X)
            ksx = self.model.kernel.forward(xstar,X)
            dK = -l**-2*(xstar-X).T*ksx
            ddK = (l**-2*var)*torch.eye(d).to(torch.float64)
        else:
            kernel_jac = functional.jacobian(self.evaluateKernel, (xstar, X), create_graph=True)
            kernel_hes = functional.hessian(self.evaluateKernel, (xstar, xstar), create_graph=True)
            dK = torch.reshape(torch.squeeze(kernel_jac[0]), (N_train, d)).T
            ddK = torch.reshape(torch.squeeze(kernel_hes[0][1]), (d, d))
        return dK, ddK

    def embed(self,xstar,jitter=1e-5):
        """ Maps from latent to data space, implements equations 2.23 - 2.24 from Rasmussen.
        We assume a different mean function across dimensions but same covariance matrix
        """
        X = self.model.X_loc
        Y = self.data
        #Y = self.model.__dict__['y'].clone().detach().t()
        n = X.shape[0]
        noise = self.model.noise_unconstrained.exp()**2
        X, Y= X.to(torch.float64), Y.to(torch.float64)
        # HACK: Do we change the type of the tensors?
        xstar = xstar.to(torch.float64) # Numerical precision needed

        Ksx = self.model.kernel.forward(xstar,X)#self.evaluateKernel(xstar,X)
        Kxx = self.model.kernel.forward(X,X) + torch.eye(n)*noise #self.evaluateKernel(X,X) + torch.eye(n)*noise
        Kss = self.model.kernel.forward(xstar,xstar)#self.evaluateKernel(xstar,xstar)
        Kinv = (Kxx).cholesky().cholesky_inverse()

        mu = Ksx.mm(Kinv).mm(Y)
        Sigma = Kss - Ksx.mm(Kinv).mm(Ksx.T) + jitter*torch.eye(xstar.shape[0]) # should be symm. positive definite
        return mu, Sigma

    def jacobian(self,xstar):
        """
        Returns the expected Jacobian at xstar

        - input
            xstar   : Points at which to compute the expected metric.
                      Size: nof_points x d
        - output
            mus     : Mean of Jacobian distribution at xstar.
                      Size: nof_points x d x D
            covs    : Covariance of Jacobian distribution at xstar (assumed independent across dimensions).
                      Size: nof_points x d x d
        """
        # QUESTION: Should the names be different? Metric is fixed and I want the names for the Jacobian to be equivalent.

        x = self.model.X_loc.double()
        #x = self.model.X_map.double()
        #x = self.model.X.double()
        sigma2 = self.model.noise_unconstrained.exp().double()**2
        N, d = x.shape
        Y = self.model.y.t().double()#__dict__['y'].clone().detach().t().double()
        D = Y.shape[1]

        # TODO: Could do private method to handle this as this will need to be chacked multiple times.
        if len(xstar.shape) <2:
            xstar = xstar.unsqueeze(0)
        nof_points =  xstar.shape[0]

        dk, ddk = self.evaluateDiffKernel_batch(xstar.double(), x)
        #kxx = self.evaluateKernel(x,x) + torch.eye(N)*sigma2
        kxx = self.model.kernel.forward(x,x) + torch.eye(N)*sigma2

        kinv = (kxx).cholesky().cholesky_inverse()
        #ksx = self.evaluateKernel(xstar.double(),x)
        ksx = self.model.kernel.forward(xstar.double(),x)

        mu_star = torch.matmul(dk.transpose(1,2),kinv.mm(Y))
        cov_star = ddk - torch.bmm(torch.matmul(dk.transpose(1,2),kinv),dk)

        # if nof_points == 1:
        #     mu_star = mu_star.squeeze()
        #     cov_star = cov_star.squeeze()
        return mu_star, cov_star

    def sample_jacobian(self,xstar,nof_samples=1):
        """
        Returns samples of Jacobian at point xstar.
        - input
            xstar           : Points...
            nof_samples     : Number of desired sampled at each point
        - output
            sample_jacobian : Size: nof_samples x nof_points x d x D.

        Note that looping is a lot faster than sampling once with a full covariance matrix!
        """

        d = self.model.X_loc.shape[1]
        Y = self.model.base_model.__dict__['y'].clone().detach().t()
        D = Y.shape[1]

        if len(xstar.shape) <2:
            xstar = xstar.unsqueeze(0)
        nof_points =  xstar.shape[0]

        mu_J,cov_J = self.jacobian(xstar)

        j_samples_by_point = list() # list of samples of each point, length: nof_samples
        for sample_index in range(nof_samples):
            j_by_point = list() # list of j by point, length: nof_points
            for point_index in range(nof_points):
                mean = mu_J[point_index,:].view(1,-1).squeeze()
                cov = kronecker(cov_J[point_index,:],torch.eye(D))
                cov  = torch.cholesky(cov) + torch.cholesky(cov.t(),upper=True) # force symmetry
                J = torch.distributions.multivariate_normal.MultivariateNormal(mean.double(), covariance_matrix=cov.double())
                sample = J.rsample()
                # TODO: Understand rsample vs sample. This has  something to do with pairwise derivatives...
                j_by_point.append(sample.reshape(mu_J.shape[1:]))
            j_samples_by_point.append(torch.stack(j_by_point).unsqueeze(0))

        sample_jacobian = torch.cat(j_samples_by_point)
        # if nof_points == 1:
        #     sample_jacobian = sample_jacobian.squeeze()
        # if nof_samples == 1:
        #     sample_jacobian = sample_jacobian.squeeze()
        return sample_jacobian

    def metric(self,xstar):
        """Computes the **expected** Riemannian metric at points xstar as the metric has to be deterministic
        - input:
            xstar: Points at which to compute the expected metric. Should be of size nof_points x d
        - output:
            G_expected: Expected metric tensor of shape nof_points x d x d
        """
        #d = self.model.X_loc.shape[1]
        d = 2 # HACK: Sort this out
        #Y = self.model.__dict__['y'].clone().detach().t()
        #Y = self.model.__dict__['y'].clone().detach().t()
        #Y = self.model.base_model.y
        Y = self.model.y
        D = Y.shape[1]
        mu_Js,cov_J = self.jacobian(xstar)
        G_expected = torch.bmm(mu_Js,mu_Js.transpose(1,2)) + D* cov_J
        return G_expected

    def sample_metric(self,xstar,nof_samples=1):
        """Allows sampling from the Riemannian metric at multiple points.
        - input:
            xstar: Points at which to compute the expected metric. Should be of size nof_points x d
            nof_samples: Number of desired samples
        - output:
            g_sample: Samples of the metric. Size (nof_points, nof_samples, d x d)
        """
        j_samples = self.sample_jacobian(xstar,nof_samples=nof_samples)
        g_sample = torch.matmul(j_samples,j_samples.transpose(2,3))
        return g_sample

    def derivatives(self,coords, method='obs_derivatives'):
        """
        Function using two methods to obtain the variance and expectation of
        the partial derivatives (df/dt) of the map f.
        df/dt = df/dc * dc/dt.
        inputs:
            - coords: coords of latent variables from the spline (N_test points x d)
            - method: method used (discretization or observational derivatives)
        output:
            - var_derivatives: variance (vector, size: (N_test-1))
            - mu_derivatives: expectation (matrix, size: (N_test-1)*D)
        """
        if method == 'discretization':
            mean_c, var_c = self.embed(coords)
            mu_derivatives = mean_c[0:-1,:] - mean_c[1:,:]
            var_derivatives = var_c.diagonal()[1:]+var_c.diagonal()[0:-1] -2*(var_c[0:-1,1:].diagonal())
            # mu, var = mu_{i+1} - mu_{i}, s_{i+1,i+1} + s_{i,i} - 2*s_{i,i+1}

        elif method == 'obs_derivatives':
            dc = coords[0:-1,:] - coords[1:,:] # derivatives of the spline (dc/dt)
            c = (coords[0:-1,:] + coords[1:,:])/2 # coordinates of derivatives
            X = self.model.X_loc
            Y = self.model.base_model.__dict__['y'].clone().detach().t()
            X, Y = X.to(torch.float64), Y.to(torch.float64) # num. precision
            c, dc = c.to(torch.float64), dc.to(torch.float64)
            noise = self.model.base_model.noise_unconstrained.exp()**2
            N, D, d, Ntest = Y.shape[0], Y.shape[1], X.shape[1], dc.shape[0]

            #kxx = self.evaluateKernel(X,X) + torch.eye(N)*noise
            kxx = self.model.kernel.forward(X,X) + torch.eye(N)*noise
            kinv = (kxx).cholesky().cholesky_inverse()
            mu_derivatives = torch.zeros(Ntest, D)
            var_derivatives = torch.zeros(Ntest)

            for nn in range(Ntest):
                dk, ddk = self.evaluateDiffKernel(c[nn,:].unsqueeze(0), X)
                var_star = ddk - dk.mm(kinv).mm(dk.T) # var(df/dc)
                var = dc[nn,:].unsqueeze(0).mm(var_star).mm(dc[nn,:].unsqueeze(0).T) # var(df/dt) = dc/dt * var(df/dc) * (dc/dt).T
                var_derivatives[nn] = var
                for dd in range(D):
                    y = Y[:,dd].unsqueeze(1)
                    mu_star = dk.mm(kinv).mm(y) # mean(df/dc)
                    mu = dc[nn,:].unsqueeze(0).mm(mu_star) # mean(df/dt) = dc/dt * mean(df/dc)
                    mu_derivatives[nn,dd] = mu

        return mu_derivatives, var_derivatives

    def curve_energy(self, coords):

        # Takes
        #n = self.model.X_loc.shape[0]
        #D = self.model.base_model.__dict__['y'].shape[0]
        D, n= self.data.shape

        mu,var = self.embed(coords) # NxD
        if self.mode == 'riemannian':
            mu, var = self.embed(coords)
            energy = (mu[1:,:] -mu[0:-1,:]).pow(2).sum() + D*(2*var.trace() - 2*var[1:,0:-1].trace())
        elif self.mode == 'finslerian':
            from .stats import NonCentralNakagami
            mu, var = self.derivatives(coords)
            non_central_nakagami = NonCentralNakagami(mu, var)
            energy = (non_central_nakagami.expectation()**2).sum()
        print('{} energy: {:.4f} \r'.format(self.mode, energy.detach().numpy()), end='\r')
        # QUESTION: Is it possible to print the last computed energy permanently? Probably not from this function.
        return energy

    def curve_length(self,sample):
        """Compute the Euclidean length of the coordinates given in sample
        - input
            sample: set of points that constitute a curve. Size nof_point x
        - output
            length: Euclidean length of curve

        """
        # QUESTION: Would we ever need to backpropagate through the length?
        length = 0
        for i in range(len(sample[:,0])-1):
            length = length + torch.norm(sample[:,i]-sample[:,i+1])
        return length

    def sample_geodesic(self,geodesic_points):
        """Function that gets a sample in dataspace from deterministic geodesic.
        Input:
            - geodesic_points: Coordinates of points that constitute the geodesic.
                Can be obtained from a _curve_ by
                    C = CubicSpline(begin=x0,end=x1,num_nodes=2,requires_grad=True)
                    t = torch.linspace(0, 1, 100)
                    geodesic_points = C(t)
        Output:
            - Coordinates of geodesic in data space
        """

        Y = self.model.base_model.__dict__['y'].clone().detach().t()
        D = Y.shape[1]

        mu,cov = self.embed(geodesic_points) # predictive GP distribution

        samples = list()
        for point_index in range(mu.shape[1]):
            mean = mu[:,point_index].view(1,-1).squeeze()
            f = torch.distributions.multivariate_normal.MultivariateNormal(mean.double(), covariance_matrix=cov.double())
            samples.append( f.rsample())
        hej = torch.stack(samples)
        return hej.T



# TODO: Notation: Use coords or xstar
# TODO: Write a test function that calls all functions.
# QUESTION: Not sure where these two function belong.
# QUESTION: Do we have a plotting module yet? Should we?
# TODO: Cilie: Add movie making to module

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def symmetrize_matrix(cov):
    symmetric_cov  = torch.cholesky(cov) + torch.cholesky(cov.t(),upper=True) # force symmetry
    return symmetric_cov

def plotCovarianceAsEllipsis(xstar, matrix,meanfl=True,precisionfl=False,eig_scale = 0.25):
    """Function that draw a covariance matrix (a metric) to get an idea about the space
    inputs:
        - xstar: Point in which the ellipsis should be plotted
        - matrix: Matrix that should define the ellipsis
        - eig_scale: A scaling of the eigenvalues such that the ellipses have a size that make sense. A total hack.
    output:
        - None. Modifies a current axis
    """

    def setProperAxes(eig_scale  = eig_scale):

        xmin =  min(geodesic_points[:,0].detach()) - eig_scale
        xmax =  max(geodesic_points[:,0].detach()) + eig_scale
        ymin =  min(geodesic_points[:,1].detach()) - eig_scale
        ymax =  max(geodesic_points[:,1].detach()) + eig_scale

        return xmin,xmax,ymin,ymax

    from matplotlib.patches import Ellipse
    xstar = tuple(xstar.squeeze().detach().numpy())

    if precisionfl:
        matrix = matrix.inverse()
    eigvals,eigvecs = torch.eig(matrix, eigenvectors=True, out=None)

    # print(eigvals)
    # quit()
    # Compute angles between x-axis and eigenvector to be able to determine angle of rotation for ellipsis
    # TODO: I don't have to do this for both, I should just do it for the max eigenvalue.
    vector_1 = eigvecs[0].detach().numpy()
    vector_2 = eigvecs[1].detach().numpy()
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    zero_vec = [1,0]
    unit_zero_vec = zero_vec / np.linalg.norm(zero_vec)

    # I think the eigenvectors are always sorted so I just compute the angle for the first.
    hest1 = np.arccos(np.dot(unit_vector_1, unit_zero_vec))/np.pi*180
    angle = np.arccos(np.dot(unit_vector_1, unit_zero_vec))/np.pi*180

    # Scale eigenvalues to be "suitable" ...
    eigval_1 = eigvals[0][0].item()
    eigval_2 = eigvals[1][0].item()
    maxeig = max(eigval_1,eigval_2)

    # HACK: Scaling eigenvalues to be able to visualise metrics.
    eigval_1_scaled = eig_scale*eigval_1/maxeig
    eigval_2_scaled = eig_scale*eigval_2/maxeig

    if meanfl:
        ec = 'r'
    else:
        ec ='b'
    e = Ellipse(xstar, eigval_1_scaled, eigval_2_scaled,angle=angle,fc = 'w',ec = ec)
    e.set_alpha(0.3)

    ax = plt.gca()
    #ax.plot(xstar[0],xstar[1],'ro')
    ax.add_artist(e)
    # TODO: Modify axes limits such that ellipses are wholly contained in the plot
    return












#
