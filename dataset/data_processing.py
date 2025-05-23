import os, yaml
import torch
import numpy as np
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader

from dataset.graph import Graph
from dataset.transform import TorsionNoiseTransform

class DataProcessing:
    def __init__(self, param: dict, reprocess: bool=True) -> None:
        self.param = param
        self.reprocess = reprocess
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = self.gen_dataset()
        self.train_dataset, self.val_dataset, self.test_dataset = self.split_dataset()
        self.train_loader, self.val_loader, self.test_loader = self.gen_loader()

    def gen_dataset(self) -> Graph:
        transform = TorsionNoiseTransform(
            sigma_min=self.param['sigma_min'],
            sigma_max=self.param['sigma_max'],
            boltzmann_weight=self.param['boltzmann_weight']
            )
        dataset = Graph(
            root=self.param['path'],
            transform=transform,
            pickle_file=self.param['pickle_file'],
            atom_type=self.param['atom_type'],
            default_node_attr=None,
            default_edge_attr=None,
            boltzmann_resampler=None,
            reprocess=self.reprocess,
            num_workers=self.param['num_workers']
        )
        with open(os.path.join(self.param['path'], f'processed/model_parameters.yml'), 'w', encoding = 'utf-8') as mp:
            yaml.dump(self.param, mp, allow_unicode=True, sort_keys=False)
        return dataset

    def split_dataset(self) -> tuple[Subset, Subset, Subset]:
        if self.param['split_method'] == 'random':
            train_size = int(self.param['train_size'] * len(self.dataset))
            val_size = int(self.param['val_size'] * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator = torch.Generator().manual_seed(self.param['seed'])
                )

        elif self.param['split_method'] == 'manual':
            indices = np.load(self.param['split_file'], allow_pickle=True)
            train_dataset = Subset(self.dataset, indices[0])
            val_dataset = Subset(self.dataset, indices[1])
            test_dataset = Subset(self.dataset, indices[2])

        else:
            raise NotImplementedError("Split method not implemented.")

        none_idx = set([i for i, data in enumerate(self.dataset.data) if data is None])
        for dataset in [train_dataset, val_dataset, test_dataset]:
            dataset.indices = list(set(dataset.indices) - none_idx)

        return train_dataset, val_dataset, test_dataset

    def gen_loader(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.param['batch_size'],
            num_workers = self.param['num_workers'],
            shuffle = True,
            pin_memory=True
            )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size = self.param['batch_size'],
            num_workers = self.param['num_workers'],
            shuffle = False,
            pin_memory=True
            )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size = self.param['batch_size'],
            num_workers = self.param['num_workers'],
            shuffle = False,
            pin_memory=True
            )

        return train_loader, val_loader, test_loader
