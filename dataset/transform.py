import torch
import random
import numpy as np
from torch_geometric.transforms import BaseTransform

from utils.torsion import modify_conformer

class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi, boltzmann_weight=False):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.boltzmann_weight = boltzmann_weight

    def __call__(self, data):
        # select conformer
        if self.boltzmann_weight:
            data.pos = random.choices(data.pos, data.weights, k=1)[0]
        else:
            data.pos = random.choice(data.pos)

        try:
            edge_mask, mask_rotate = data.edge_mask, data.mask_rotate
        except:
            edge_mask, mask_rotate = data.mask_edges, data.mask_rotate
            data.edge_mask = torch.tensor(data.mask_edges)

        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        data.node_sigma = sigma * torch.ones(data.num_nodes)

        torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge_mask.sum())
        data.pos = modify_conformer(data.pos, data.edge_index.T[edge_mask], mask_rotate, torsion_updates)
        data.edge_rotate = torch.tensor(torsion_updates)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')