import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from typing import Any, Literal
from torch_geometric.data import Data


dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
# qm9_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
# drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
#                'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
#                'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
#                'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}

def one_k_encoding(value: Any, choices: list) -> list[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def featurize_mol(mol: Chem.rdchem.Mol, atom_type: list) -> Data:
    """
    Part of the featurisation code taken from GeoMol https://github.com/PattanaikL/GeoMol
    Returns:
        x: node features
        z: atomic numbers of the nodes (the symbol one hot is included in x)
        edge_index: [2, E] tensor of node indices forming edges
        edge_attr: edge features
    """

    N = mol.GetNumAtoms()
    atomic_number = []
    atom_features = []
    # chiral_tag = []
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        atom: Chem.rdchem.Atom
        atom_features.extend(one_k_encoding(atom.GetSymbol(), atom_type))
        # chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend([atom.GetAtomicNum(),
                              1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2]))
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        bond: Chem.rdchem.Bond
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    x = torch.tensor(atom_features).view(N, -1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z=z)

def featurize_mol_from_smiles(smiles: str, atom_type: list) -> tuple[None, None] | tuple[Chem.rdchem.Mol, Data]:
    # filter fragments
    if '.' in smiles:
        return None, None

    # filter mols rdkit can't intrinsically handle
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
    else:
        return None, None
    N = mol.GetNumAtoms()

    # filter out mols model can't make predictions for
    if not mol.HasSubstructMatch(dihedral_pattern):
        return None, None
    if N < 4:
        return None, None

    data = featurize_mol(mol, atom_type)
    data.name = smiles
    return mol, data
