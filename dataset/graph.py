import torch
import numpy as np
import os, tqdm, shutil, pickle
from collections import defaultdict
from pathos.multiprocessing import Pool
from typing import Callable, List, Optional, Dict
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from copy import deepcopy

from dataset.featurization import featurize_mol, dihedral_pattern
from utils.torsion import get_transformation_mask

class Graph(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable]=None,
        pickle_file: str=None,
        atom_type: List[str]=None,
        default_node_attr: Dict=None,
        default_edge_attr: Dict=None,
        boltzmann_resampler=None,
        reprocess: bool=False,
        num_workers: int=1,
        ) -> None:
        self.root = root
        self.pickle_file = pickle_file
        self.atom_type = atom_type
        self.default_node_attr = default_node_attr
        self.default_edge_attr = default_edge_attr
        self.boltzmann_resampler = boltzmann_resampler
        self.reprocess = reprocess
        self.num_workers = num_workers
        if self.reprocess:
            self._reprocess()
        super().__init__(root, transform)
        with open(self.processed_paths[0], 'rb') as f:
            self.data = pickle.load(f)

    @property
    def raw_file_names(self) -> List[str]:
        '''define default path to input data files.
        '''
        return ['merged_mol.sdf', 'smiles.csv']

    @property
    def processed_file_names(self) -> str:
        '''define default path to processed data file.
        '''
        return 'graph_data.pkl'

    def _reprocess(self):
        if os.path.exists(os.path.join(self.root, 'processed/')):
            shutil.rmtree(os.path.join(self.root, 'processed/'))

    def process(self):
        '''process raw data to generate dataset
        '''
        self.failures = defaultdict(int)
        with open(self.pickle_file, 'rb') as pf:
            db: dict = pickle.load(pf)
        with Pool(self.num_workers) as pool:
            datapoints = list(tqdm.tqdm(pool.imap(self.featurize_conformer, db.values()), total=len(db)))

        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(datapoints, f)

    def filter_smiles(
        self,
        smiles: str,
        ) -> bool:
        if '.' in smiles:
            self.failures['dot_in_smiles'] += 1
            return False

        # filter mols rdkit can't intrinsically handle
        mol: Chem.rdchem.Mol = Chem.MolFromSmiles(smiles)
        if not mol:
            self.failures['mol_from_smiles_failed'] += 1
            return False

        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            self.failures['no_substruct_match'] += 1
            return False

        if N < 4:
            self.failures['mol_too_small'] += 1
            return False
        return True

    def filter_data(
        self,
        data: Data,
        ) -> Data | None:
        edge_mask, mask_rotate = get_transformation_mask(data)
        if np.sum(edge_mask) < 0.5:
            self.failures['no_rotable_bonds'] += 1
            return None

        data.edge_mask = torch.tensor(edge_mask)
        data.mask_rotate = mask_rotate
        return data

    def featurize_conformer(self, mol_dict: dict):
        confs = mol_dict['conformers']
        smiles = mol_dict["smiles"]
        if not self.filter_smiles(smiles):
            return None

        mol_ = Chem.MolFromSmiles(smiles)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)

        pos = []
        weights = []
        for conf in confs:
            mol: Chem.rdchem.Mol = conf['rd_mol']

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                print(e)
                continue

            if conf_canonical_smi != canonical_smi:
                continue

            pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
            weights.append(conf['boltzmannweight'])
            correct_mol = mol

            if self.boltzmann_resampler is not None:
                # torsional Boltzmann generator uses only the local structure of the first conformer
                break

        # return None if no non-reactive conformers were found
        if len(pos) == 0:
            self.failures['featurize_mol_failed'] += 1
            return None

        data = featurize_mol(correct_mol, self.atom_type)
        normalized_weights = list(np.array(weights) / np.sum(weights))
        if np.isnan(normalized_weights).sum() != 0:
            print(smiles, len(confs), len(pos), weights)
            normalized_weights = [1 / len(weights)] * len(weights)
        data.canonical_smi, data.mol, data.pos, data.weights = canonical_smi, correct_mol, pos, normalized_weights
        data = self.filter_data(data)

        return data

    def len(self):
        return len(self.data)

    def get(self, idx):
        data = self.data[idx]
        if self.boltzmann_resampler:
            self.boltzmann_resampler.try_resample(data)
        return deepcopy(data)
