#!/usr/bin/env python3
import torch
from torch.autograd import grad
from .curve import *
from .geodesics import *
from abc import ABC, abstractmethod


class __Dist2__(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, p0, p1, C=None, A=None):
        with torch.no_grad():
            with torch.enable_grad():
                # TODO: Only perform the computations needed for backpropagation (check if p0.requires_grad and p1.requires_grad)
                if C is None:
                    print('Calculating Curve...')
                    C, success = M.connecting_geodesic(p0, p1)

                lm0 = C.deriv(torch.zeros(1)).squeeze(1)  # log(p0, p1); # Bx(d) - This is v(0)
                lm1 = -C.deriv(torch.ones(1)).squeeze(1)  # log(p1, p0); # Bx(d) - This is v(1)
                M0 = M.metric(p0)  # Bx(d)x(d) or Bx(d)
                M1 = M.metric(p1)  # Bx(d)x(d) or Bx(d)
                if M0.ndim == 3:  # metric is square
                    Mlm0 = M0.bmm(lm0.unsqueeze(-1)).squeeze(-1)  # Bx(d)  # bmm: batch matmul
                    Mlm1 = M1.bmm(lm1.unsqueeze(-1)).squeeze(-1)  # Bx(d)  # bmm: batch matmul
                else:
                    Mlm0 = M0 * lm0  # Bx(d)
                    Mlm1 = M1 * lm1  # Bx(d)

                if A is None:
                    retval = (lm0 * Mlm0).sum(dim=-1)  # B
                else:
                    v_TA_T = torch.transpose(lm0.unsqueeze(-1), 1, 2).matmul(A.T)  # [batch, 1, d]
                    v_TA_TM = v_TA_T.bmm(M0)
                    Av = A.unsqueeze(0).repeat([lm0.shape[0], 1, 1]).bmm(lm0.unsqueeze(-1))
                    retval = v_TA_TM.bmm(Av).squeeze(-1)  # [batch, 1]

                ctx.save_for_backward(Mlm0, Mlm1, lm0, lm1, A, M0, M1)

        return retval

    @staticmethod
    def backward(ctx, grad_output):
        Mlm0, Mlm1, lm0, lm1, A, M0, M1 = ctx.saved_tensors

        if A is None:
            return (None,
                    2.0 * grad_output.view(-1, 1) * Mlm0,
                    2.0 * grad_output.view(-1, 1) * Mlm1,
                    None,
                    None)
        else:
            covariance_inv = A.T @ A
            A_batched = A.unsqueeze(0).repeat([M0.shape[0], 1, 1])
            A_T_batched = A.T.unsqueeze(0).repeat([M0.shape[0], 1, 1])

            # p0 # -2A^TAMv
            inv_cov_Mlm0 = covariance_inv.unsqueeze(0).repeat([Mlm0.shape[0], 1, 1]).bmm(Mlm0.unsqueeze(-1)).squeeze(-1)
#             inv_cov_Mlm0 = A_T_batched.bmm(M0).bmm(A_batched).bmm(lm0.unsqueeze(-1)).squeeze(-1)  # -2A^TMAv

            # p1 # -2A^TAMv
            inv_cov_Mlm1 = covariance_inv.unsqueeze(0).repeat([Mlm1.shape[0], 1, 1]).bmm(Mlm1.unsqueeze(-1)).squeeze(-1)
#             inv_cov_Mlm1 = A_T_batched.bmm(M1).bmm(A_batched).bmm(lm1.unsqueeze(-1)).squeeze(-1)  # -2A^TMAv

            # covariance gradients.
            AMv = A_batched.bmm(M0).bmm(lm0.unsqueeze(-1))
            v_TM_T = torch.transpose(lm0.unsqueeze(-1), 1, 2).bmm(torch.transpose(M0, 1, 2))
            grad_cov = AMv.bmm(v_TM_T)

            return (None,
                    -2 * inv_cov_Mlm0 * grad_output.view(-1, 1),
                    -2 * inv_cov_Mlm1 * grad_output.view(-1, 1),
                    None,
                    2 * grad_cov * grad_output.view(-1, 1, 1))


class Manifold(ABC):
    """
    A common interface for manifolds. Specific manifolds should inherit
    from this abstract base class abstraction.
    """

    def curve_energy(self, curve):
        """
        Compute the discrete energy of a given curve.

        Input:
            curve:      a Nx(d) torch Tensor representing a curve or
                        a BxNx(d) torch Tensor representing B curves.

        Output:
            energy:     a scalar corresponding to the energy of
                        the curve (sum of energy in case of multiple curves).
                        It should be possible to backpropagate through
                        this in order to compute geodesics.

        Algorithmic note:
            The default implementation of this function rely on the 'inner'
            function, which in turn call the 'metric' function. For some
            manifolds this can be done more efficiently, in which case it
            is recommended that the default implementation is replaced.
        """
        if curve.dim() == 2:
            curve.unsqueeze_(0) # add batch dimension if one isn't present
        # Now curve is BxNx(d)
        d = curve.shape[2]
        delta = curve[:, 1:] - curve[:, :-1] # Bx(N-1)x(d)
        flat_delta = delta.view(-1, d) # (B*(N-1))x(d)
        energy = self.inner(curve[:, :-1].view(-1, d), flat_delta, flat_delta) # B*(N-1)
        return energy.sum() # scalar

    def curve_length(self, curve):
        """
        Compute the discrete length of a given curve.

        Input:
            curve:      a Nx(d) torch Tensor representing a curve or
                        a BxNx(d) torch Tensor representing B curves.

        Output:
            length:     a scalar or a B element Tensor containing the length of
                        the curve.

        Algorithmic note:
            The default implementation of this function rely on the 'inner'
            function, which in turn call the 'metric' function. For some
            manifolds this can be done more efficiently, in which case it
            is recommended that the default implementation is replaced.
        """
        if curve.dim() == 2:
            curve.unsqueeze_(0) # add batch dimension if one isn't present
        # Now curve is BxNx(d)
        B, N, d = curve.shape
        delta = curve[:, 1:] - curve[:, :-1] # Bx(N-1)x(d)
        flat_delta = delta.view(-1, d) # (B*(N-1))x(d)
        energy = self.inner(curve[:, :-1].view(-1, d), flat_delta, flat_delta) # B*(N-1)
        length = energy.view(B, N-1).sqrt().sum(dim=1) # B
        return length

    @abstractmethod
    def metric(self, points):
        """
        Return the metric tensor at a specified set of points.

        Input:
            points:     a Nx(d) torch Tensor representing a set of
                        points where the metric tensor is to be
                        computed.

        Output:
            M:          a Nx(d)x(d) or Nx(d) torch Tensor representing
                        the metric tensor at the given points.
                        If M is Nx(d)x(d) then M[i] is a (d)x(d) symmetric
                        positive definite matrix. If M is Nx(d) then M[i]
                        is to be interpreted as the diagonal elements of
                        a (d)x(d) diagonal matrix.
        """
        pass

    def inner(self, base, u, v, return_metric=False):
        """
        Compute the inner product between tangent vectors u and v at
        base point.

        Mandatory inputs:
            base:       a Nx(d) torch Tensor representing the points of
                        tangency corresponding to u and v.
            u:          a Nx(d) torch Tensor representing N tangent vectors
                        in the tangent spaces of 'base'.
            v:          a Nx(d) torch Tensor representing tangent vectors.

        Optional input:
            return_metric:  if True, the metric at 'base' is returned as a second
                            output. Otherwise, only one output is provided.

        Output:
            dot:        a N element torch Tensor containing the inner product
                        between u and v according to the metric at base.
            M:          if return_metric=True this second output is also
                        provided. M is a Nx(d)x(d) or a Nx(d) torch Tensor
                        representing the metric tensor at 'base'.
        """
        M = self.metric(base) # Nx(d)x(d) or Nx(d)
        diagonal_metric = M.dim() == 2
        if diagonal_metric:
            dot = (u * M * v).sum(dim=1) # N
        else:
            Mv = M.bmm(v.unsqueeze(-1)) # Nx(d)
            dot = u.unsqueeze(1).bmm(Mv).flatten() # N    #(u * Mv).sum(dim=1) # N
        if return_metric:
            return dot, M
        else:
            return dot

    def volume(self, points):
        """
        Evaluate the volume measure at a set of given points.

        Input:
            points:     a Nx(d) torch Tensor representing points on
                        the manifold.

        Output:
            vol:        a N element torch Tensor containing the volume
                        element at each point.

        Algorithmic note:
            The algorithm merely compute the square root determinant of
            the metric at each point. This may be expensive and may be numerically
            unstable; if possible, you should use the 'log_volume' function
            instead.
        """
        M = self.metric(points) # Nx(d)x(d) or Nx(d)
        diagonal_metric = M.dim() == 2
        if diagonal_metric:
            vol = M.prod(dim=1).sqrt() # N
        else:
            vol = M.det().sqrt() # N
        return vol

    def log_volume(self, points):
        """
        Evaluate the logarithm of the volume measure at a set of given points.

        Input:
            points:     a Nx(d) torch Tensor representing points on
                        the manifold.

        Output:
            log_vol:    a N element torch Tensor containing the logarithm
                        of the volume element at each point.

        Algorithmic note:
            The algorithm merely compute the log-determinant of the metric and
            divide by 2. This may be expensive.
        """
        M = self.metric(points) # Nx(d)x(d) or Nx(d)
        diagonal_metric = M.dim() == 2
        if diagonal_metric:
            log_vol = 0.5 * M.log().sum(dim=1) # N
        else:
            log_vol = 0.5 * M.logdet() # N
        return log_vol

    def geodesic_system(self, c, dc):
        """
        Evaluate the geodesic ODE of the manifold.

        Inputs:
            c:          a Nx(d) torch Tensor representing a set of points
                        in latent space (e.g. a curve).
            dc:         a Nx(d) torch Tensor representing the velocity at
                        the points.

        Output:
            ddc:        a Nx(d) torch Tensor representing the second temporal
                        derivative of the geodesic passing through c with
                        velocity dc.

        Algorithmic notes:
            The algorithm evaluates the equation
                c'' = M^{-1} * (0.5 * dL/dc - dM/dt * c')
                L = c'^T * M * c'
            The term dL/dc is evaluated with automatic differentiation, which
            imply a loop over N, which can be slow. The derivative dM/dt is
            evaluated using finite differences, which is fast but may imply
            a slight loss of accuracy.
            When possible, it may be beneficial to provide a specialized version
            of this function.
        """
        N, d = c.shape
        requires_grad = c.requires_grad or dc.requires_grad

        # Compute dL/dc using auto diff
        z = c.clone().requires_grad_() # Nx(d)
        dz = dc.clone().requires_grad_() # Nx(d)
        L, M = self.inner(z, dz, dz, return_metric=True) # N, Nx(d)x(d) or N, Nx(d)
        if requires_grad:
            dLdc = torch.cat([grad(L[n], z, create_graph=True)[0][n].unsqueeze(0) for n in range(N)]) # Nx(d)
        else:
            dLdc = torch.cat([grad(L[n], z, retain_graph=(n < N-1))[0][n].unsqueeze(0) for n in range(N)]) # Nx(d)

        # Use finite differences to approximate dM/dt as that is more
        # suitable for batching.
        # TODO: make this behavior optional allowing exact expressions.
        #h = 1e-4
        #with torch.set_grad_enabled(requires_grad):
        #    dMdt = (self.metric(z + h*dz) - M) / h # Nx(d)x(d) or Nx(d)
        #print('fd', dMdt, dMdt.shape)

        M = self.metric(z)
        diagonal_metric = M.dim() == 2
        if requires_grad:
            if diagonal_metric:
                dMdt = torch.tensor([[torch.sum(grad(M[n, i], z, create_graph=True)[0] * dz) for i in range(d)] for n in range(N)]) # Nx(d)
            else:
                dMdt = torch.tensor([[torch.sum(grad(M[n, i, j], z, create_graph=True)[0] * dz) for i in range(d) for j in range(d)] for n in range(N)]).view(N, d, d) # Nx(d)x(d) # TODO: figure out how to not store the graph
        else:
            if diagonal_metric:
                dMdt = torch.tensor([[torch.sum(grad(M[n, i], z, retain_graph=True)[0] * dz) for i in range(d)] for n in range(N)]) # Nx(d) # TODO: figure out how to not store the graph
            else:
                dMdt = torch.tensor([[torch.sum(grad(M[n, i, j], z, retain_graph=True)[0] * dz) for i in range(d) for j in range(d)] for n in range(N)]).view(N, d, d) # Nx(d)x(d) # TODO: figure out how to not store the graph
        dMdt = dMdt.to(dz.device)
        #print('ad', dMdt, dMdt.shape)

        # Evaluate full geodesic ODE:
        # c'' = (0.5 * dL/dc - dM/dt * c') / M
        with torch.set_grad_enabled(requires_grad):
            if diagonal_metric:
                ddc = (0.5 * dLdc - dMdt * dz) / M # Nx(d)
            else:
                # XXX: Consider Cholesky-based solver
                Mddc = 0.5 * dLdc - dMdt.bmm(dz.unsqueeze(-1)).squeeze(-1) # Nx(d)
                #print("p", z.squeeze(0).data.tolist(), "eigenvalues:",
                #      round(M.squeeze(0).eig()[0][0][0].item(), 2),
                #      round(M.squeeze(0).eig()[0][1][0].item(),2))
                #print("inversed matrix:", torch.inverse(M.squeeze(0)).data.tolist())
                ddc, _ = torch.solve(Mddc.unsqueeze(-1), M) # Nx(d)x1
                ddc = ddc.squeeze(-1) # Nx(d)
        return ddc

    def connecting_geodesic(self, p0, p1, init_curve=None):
        """
        Compute geodesic connecting two points.

        Mandatory inputs:
            p0:         a torch Tensor representing the initial point
                        of the requested geodesic.
            p1:         a torch Tensor representing the end point
                        of the requested geodesic.

        Optional input:
            init_curve: a curve representing an initial guess of the
                        requested geodesic. If the end-points of the
                        initial curve do not correspond to p0 and p1,
                        then the curve is modified accordingly.
                        If None then the default constructor of the
                        chosen curve family is applied.
                        Default: None
        """
        if init_curve is None:
            curve = CubicSpline(p0, p1, device=p0.device)
        else:
            curve = init_curve
            curve.begin = p0
            curve.end = p1

        #success = geodesic_minimizing_energy_sgd(curve, self)
        success = geodesic_minimizing_energy(curve, self)
        return (curve, success)

    def shooting_geodesic(self, p, v, t=torch.linspace(0, 1, 50), requires_grad=False):
        """
        Compute the geodesic with a given starting point and initial velocity.

        Mandatory inputs:
            p:              a torch Tensor with D elements representing the initial
                            position on the manifold of the requested geodesic.
            v:              a torch Tensor with D elements representing the initial
                            velocity of the requested geodesic.

        Optional inputs:
            t:              a torch Tensor of time values where the requested geodesic
                            will be computed. This must at least contain two values
                            where the first must be 0.
                            Default: torch.linspace(0, 1, 50)
            requires_grad:  if True it is possible to backpropagate through this
                            function.
                            Default: False

        Output:
            c:              a torch Tensor of size TxD containing points along the
                            geodesic at the reequested times.
            dc:             a torch Tensor of size TxD containing the curve derivatives
                            at the requested times.
        """
        return shooting_geodesic(self, p, v, t, requires_grad)

    def logmap(self, p0, p1, curve=None, optimize=True):
        """
        Compute the logarithm map of the geodesic from p0 to p1.

        Mandatory inputs:
            p0:         a torch Tensor representing the base point
                        of the logarithm map.
            p1:         a torch Tensor representing the end point
                        of the underlying geodesic.

        Optional inputs:
            curve:      an initial estimate of the geodesic from
                        p0 to p1.
                        Default: None
            optimize:   if False and an initial curve is present, then
                        the initial curve is assumed to be the true
                        geodesic and the logarithm map is extracted
                        from the initial curve.
                        Default: True

        Output:
            lm:         a torch Tensor with D elements representing
                        a tangent vector at p0. The norm of lm
                        is the geodesic distance from p0 to p1.
        """
        if curve is None:
            curve = self.connecting_geodesic(p0, p1, init_curve=None)[0]
        elif curve is not None and optimize:
            curve = self.connecting_geodesic(p0, p1, init_curve=curve)[0]
        with torch.no_grad():
            lm = curve.deriv(torch.zeros(1))
        return lm, curve

    def expmap(self, p, v, t=torch.linspace(0, 1, 5)):
        """
        Compute the exponential map starting at p with velocity v.

        Mandatory inputs:
            p:          a torch Tensor representing the base point
                        of the exponential map.
            v:          a torch Tensor representing the velocity of
                        the underlying geodesic at p.

        Output:
            u:          a torch Tensor corresponding to the end-point
                        of the requested geodesic.

        Algorithmic note:
            This implementation use a numerical ODE solver to integrate
            the geodesic ODE. This in turn require evaluating both the
            metric and its derivatives, which may be expensive.
        """
        requires_grad = p.requires_grad or v.requires_grad
        c, _ = shooting_geodesic(self, p, v, t,
                                 requires_grad=requires_grad)
        return c, c[-1].view(1, -1)

    def dist2(self, p0, p1, A=None): # XXX: allow for warm-starting the geodesic
        """
        Compute the squared geodesic distance between two points.

        Mandatory inputs:
            p0: a torch Tensor representing one point.
            p1: a torch Tensor representing another point.

        Output:
            d2: the squared geodesic distance between the two
                given points.
        """

        d2 = __Dist2__()
        return d2.apply(self, p0, p1)

    def dist2_explicit(self, p0, p1, C=None, A=None):
        """
        Compute the squared geodesic distance between two points.
        Also returns Curve and success

        Mandatory inputs:
            p0: a torch Tensor representing one point.
            p1: a torch Tensor representing another point.

        Output:
            d2: the squared geodesic distance between the two
                given points.
        """
        C, success = self.connecting_geodesic(p0, p1, init_curve=C)
        d2 = __Dist2__()

        return d2.apply(self, p0, p1, C, A), C, success


class EmbeddedManifold(Manifold, ABC):
    """
    A common interface for embedded manifolds. Specific embedded manifolds
    should inherit from this abstract base class abstraction.
    """

    def curve_energy(self, curve, dt=None):
        """
        Compute the discrete energy of a given curve.

        Input:
            curve:      a Nx(d) torch Tensor representing a curve or
                        a BxNx(d) torch Tensor representing B curves.

        Output:
            energy:     a scalar corresponding to the energy of
                        the curve (sum of energy in case of multiple curves).
                        It should be possible to backpropagate through
                        this in order to compute geodesics.

        Algorithmic note:
            The algorithm rely on the deterministic embedding of the manifold
            rather than the metric. This is most often more efficient.
        """
        if curve.dim() == 2:
            curve.unsqueeze_(0)  # add batch dimension if one isn't present
        if dt is None:
            dt = (curve.shape[1] - 1)
        # Now curve is BxNx(d)
        emb_curve = self.embed(curve)  # BxNxD
        B, N, D = emb_curve.shape
        delta = emb_curve[:, 1:, :] - emb_curve[:, :-1, :]  # Bx(N-1)xD
        energy = (delta ** 2).sum((1, 2)) * dt  # B
        return energy

    def curve_length(self, curve, dt=None):
        """
        Compute the discrete length of a given curve.

        Input:
            curve:      a Nx(d) torch Tensor representing a curve or
                        a BxNx(d) torch Tensor representing B curves.

        Output:
            length:     a scalar or a B element Tensor containing the length of
                        the curve.

        Algorithmic note:
            The default implementation of this function rely on the 'inner'
            function, which in turn call the 'metric' function. For some
            manifolds this can be done more efficiently, in which case it
            is recommended that the default implementation is replaced.
        """
        if curve.dim() == 2:
            curve.unsqueeze_(0)  # add batch dimension if one isn't present
        if dt is None:
            dt = 1.0  # (curve.shape[1]-1)
        # Now curve is BxNx(d)
        emb_curve = self.embed(curve)  # BxNxD
        delta = emb_curve[:, 1:] - emb_curve[:, :-1]  # Bx(N-1)xD
        speed = delta.norm(dim=2)  # Bx(N-1)
        lengths = speed.sum(dim=1) * dt  # B
        return lengths

    def metric(self, points):
        """
        Return the metric tensor at a specified set of points.

        Input:
            points:     a Nx(d) torch Tensor representing a set of
                        points where the metric tensor is to be
                        computed.

        Output:
            M:          a Nx(d)x(d) or Nx(d) torch Tensor representing
                        the metric tensor at the given points.
                        If M is Nx(d)x(d) then M[i] is a (d)x(d) symmetric
                        positive definite matrix. If M is Nx(d) then M[i]
                        is to be interpreted as the diagonal elements of
                        a (d)x(d) diagonal matrix.
        """
        _, J = self.embed(points, jacobian=True)  # NxDx(d)
        M = J.transpose(2, 1).bmm(J) # torch.einsum("bji,bjk->bik", J, J)
        return M

    @abstractmethod
    def embed(self, points, jacobian=False):
        """
        XXX: Write me! Don't forget batching!
        """
        pass
