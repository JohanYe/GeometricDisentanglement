#!/usr/bin/env python3
import torch
import numpy as np

class Sphere:
    def __init__(self, device=None):
        self.device = device

    def intrinsic_coordinates(self, points): # encoder
        # points: NxD
        coords = torch.stack((torch.acos(points[:, 2]), torch.atan2(points[:, 1], points[:, 0]))).transpose(0, 1)
        return coords

    def embed(self, coords):  # decoder
        # coords: Nx(D-1)
        points = torch.stack((torch.sin(coords[:, 0]) * torch.cos(coords[:, 1]),
                              torch.sin(coords[:, 0]) * torch.cos(coords[:, 1]),
                              torch.cos(coords[:, 0]))).transpose(0, 1)
        return points

    def metric(self, coords):
        # coords: Nx(D-1)
        return

    def curve_energy(self, coords):
        # coords: Nx(D-1)
        points = self.embed(coords) # NxD
        energy = (points[1:] - points[:-1]).pow(2).sum()
        return energy
