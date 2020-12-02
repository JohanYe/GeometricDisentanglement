#!/usr/bin/env python3
import torch
import numpy as np
from geoml import CubicSpline
from geoml.geodesics import geodesic_minimizing_energy
import sys

class HyperSphere:
    def __init__(self, device=None, dimension=3):
        self.device = device
        self.dim = dimension

    def intrinsic_coordinates(self, points):
        # points: NxD
        raise NotImplementedError('Not implemented')
    
    def embed(self, coords):
        # Coodinates defined in disk x1**2+x2**2+...+xn**2 < 1
        # where n=self.dimension

        N = coords.shape[0]
        squares = torch.mm(coords.pow(2), torch.ones((self.dim,1)))
        last_col = torch.sqrt(torch.ones((N,1)) - squares)

        points = torch.cat((coords, last_col),dim=1)
        return points
        
    def metric(self, coords):
        # coords: Nx(D-1)
        return

    def curve_energy(self, coords):
        # coords: Nx(D-1)
        points = self.embed(coords) # NxD
        energy = (points[1:] - points[:-1]).pow(2).sum()
        return energy
   
    def geodesic_score(self, curve, p0, p1, eval_grid=200):
        alpha = torch.linspace(0, 1, eval_grid).reshape((-1, 1))
        points = self.embed(curve(alpha))
        v0 = self.embed(p0)
        v1 = self.embed(p1)
        v1 = v1 - torch.mm(v0, v1.transpose(0, 1))*v0
        if float(v1.norm()) < 1e-5:
            raise ValueError('Start and end points of geodesic too close.')
        v1 = v1/v1.norm()
        proj = torch.mm(torch.mm(points, v0.transpose(0, 1)), v0) + torch.mm(torch.mm(points, v1.transpose(0, 1)), v1)
        score = ((proj-points).pow(2).sum()/float(eval_grid))
        return score

    def random_test(self):
        a = torch.randn(1, self.dim + 1)
        b = torch.randn(1, self.dim + 1)
        # move start and end points p0 and p1 away from the boundary
        # defined by ||p0||=1, ||p1||=1 
        a = a/(a.norm()+0.5)
        b = b/(b.norm()+0.5)
        p0 = a[:,0:-1]
        p1 = b[:,0:-1]
        
        C = CubicSpline(begin=p0, end=p1, num_nodes=8, requires_grad=True)
        geodesic_minimizing_energy(C, self)
        try:
            return self.geodesic_score(C, p0, p1)
        except(ValueError):
            print("Warning: start and end points too close.")
            return

    def run_tests(self, numtests, eps=1e-5):
        for ii in range(numtests):
            out = self.random_test()
            if out is not None and (out > eps):
                print("Tests failed.")
                return 0
        return 1
