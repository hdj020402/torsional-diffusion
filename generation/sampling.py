import random, os, pickle
from dataset.featurization import featurize_mol, featurize_mol_from_smiles
from model.score_model import TensorProductScoreModel
from utils.torsion import *
import torch, copy
from copy import deepcopy
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

from utils.visualise import PDBFile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
still_frames = 10


def try_mmff(mol: Chem.rdchem.Mol) -> bool:
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        return True
    except Exception as e:
        return False


def get_seed(
    smi: str,
    seed_confs: dict[str, list] | None,
    atom_type: list
    ) -> tuple[None, None] | tuple[Chem.rdchem.Mol, Data | None]:
    if seed_confs:
        if smi not in seed_confs:
            print("smile not in seeds", smi)
            return None, None
        mol = seed_confs[smi][0]
        data = featurize_mol(mol, atom_type)
    else:
        mol, data = featurize_mol_from_smiles(smi, atom_type)
        if not mol:
            return None, None
    data.edge_mask, data.mask_rotate = get_transformation_mask(data)
    data.edge_mask = torch.tensor(data.edge_mask)
    return mol, data


def embed_seeds(
    mol: Chem.rdchem.Mol,
    data: Data,
    n_confs: int,
    single_conf: bool=False,
    smi: str | None=None,
    seed_confs: dict[str, list] | None=None,
    apply_pdb: bool=False,
    mmff=False
    ) -> tuple[list, None] | tuple[list[Data], PDBFile | None]:
    if not seed_confs:
        embed_num_confs = n_confs if not single_conf else 1
        try:
            _ = AllChem.EmbedMultipleConfs(mol, numConfs=embed_num_confs, numThreads=5)
        except Exception as e:
            print(e.output)
            pass
        if len(mol.GetConformers()) != embed_num_confs:
            print(len(mol.GetConformers()), '!=', embed_num_confs)
            return [], None
        if mmff: try_mmff(mol)

    pdb = PDBFile(mol) if apply_pdb else None
    conformers = []
    for i in range(n_confs):
        data_conf = copy.deepcopy(data)
        if single_conf:
            seed_mol = copy.deepcopy(mol)
        elif seed_confs:
            seed_mol = random.choice(seed_confs[smi])
        else:
            seed_mol = copy.deepcopy(mol)
            [seed_mol.RemoveConformer(j) for j in range(n_confs) if j != i]

        data_conf.pos = torch.from_numpy(seed_mol.GetConformers()[0].GetPositions()).float()
        data_conf.seed_mol = copy.deepcopy(seed_mol)
        if pdb:
            pdb.add(data_conf.pos, part=i, order=0, repeat=still_frames)
            if seed_confs:
                pdb.add(data_conf.pos, part=i, order=-2, repeat=still_frames)
            pdb.add(torch.zeros_like(data_conf.pos), part=i, order=-1)

        conformers.append(data_conf)
    if mol.GetNumConformers() > 1:
        [mol.RemoveConformer(j) for j in range(n_confs) if j != 0]
    return conformers, pdb


def perturb_seeds(conformers: list[Data], pdb: PDBFile=None) -> list[Data]:
    for i, data_conf in enumerate(conformers):
        torsion_updates = np.random.uniform(low=-np.pi,high=np.pi, size=data_conf.edge_mask.sum())
        data_conf.pos = modify_conformer(data_conf.pos, data_conf.edge_index.T[data_conf.edge_mask],
                                         data_conf.mask_rotate, torsion_updates)
        data_conf.total_perturb = torsion_updates
        if pdb:
            pdb.add(data_conf.pos, part=i, order=1, repeat=still_frames)
    return conformers


def sample(
    conformers: list[Data],
    model: TensorProductScoreModel,
    sigma_max: float=np.pi,
    sigma_min: float=0.01 * np.pi,
    steps: int=20,
    batch_size: int=32,
    ode: bool=False,
    pdb: PDBFile | None=None
    ):
    """
    Performs diffusion sampling to generate conformers without Prior Guidance.

    Args:
        conformers (list): Initial list of conformers (PyG Data objects).
        model (torch.nn.Module): Trained diffusion model.
        sigma_max (float): Maximum noise level for sampling.
        sigma_min (float): Minimum noise level for sampling.
        steps (int): Number of denoising steps.
        batch_size (int): Number of conformers to process in parallel.
        ode (bool): Whether to use probability flow ODE (True) or SDE (False).
        likelihood (str or None): Method for likelihood computation ('full', 'hutch', or None).
        pdb (object or None): Object for recording PDB frames during sampling (e.g., PDBWriter).

    Returns:
        list: List of generated conformers (PyG Data objects).
    """
    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)

    # Generates log-uniform sigma schedule
    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps # Time step size

    for batch_idx, data in enumerate(loader):
        # Move data batch to the appropriate device (GPU or CPU)
        data_gpu = copy.deepcopy(data).to(model.device) # Assuming model has a .device attribute or infer from model.parameters()...device

        # Iterate through each noise level
        for sigma_idx, sigma in enumerate(sigma_schedule):

            # Set the current noise level for the model input
            data_gpu.node_sigma = sigma * torch.ones(data_gpu.num_nodes, device=model.device)

            # Model prediction step (get the score/gradient)
            with torch.no_grad():
                data_gpu = model(data_gpu)

            # Calculate g factor (related to the diffusion coefficient)
            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            # Sample random noise for SDE
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape).to(model.device) # Ensure z is on the correct device
            # Get the predicted score from the model
            score = data_gpu.edge_pred # Model output (predicted gradient)

            # Calculate the perturbation/update step based on ODE or SDE
            if ode:
                # ODE step uses only the score
                perturb = 0.5 * g ** 2 * eps * score
            else:
                # SDE step uses score and random noise
                perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            # Apply the calculated perturbation to the conformers' torsion angles and update positions
            # perturb needs to be on CPU for apply_torsion_and_update_pos which likely works on numpy arrays
            conf_dataset.apply_torsion_and_update_pos(data, perturb.cpu().numpy())
            # Update the positions on the GPU data object for the next model call
            data_gpu.pos = data.pos.to(model.device)

            if pdb:
                 for conf_idx in range(data.num_graphs):
                     # Extract coordinates for the current conformer batch
                     coords = data.pos[data.ptr[conf_idx]:data.ptr[conf_idx + 1]]
                     # Add frames to PDB object (still_frames might need to be defined or passed if used here)
                     # Assuming still_frames logic from original code, or simplify
                     num_frames = 1 # Default to 1 frame per step
                     # if sigma_idx == steps - 1: num_frames = still_frames # Example if last step is recorded more
                     pdb.add(coords, part=batch_size * batch_idx + conf_idx, order=sigma_idx + 2, repeat=num_frames)

    return conformers


def pyg_to_mol(
    mol: Chem.rdchem.Mol,
    data: Data,
    mmff: bool=False,
    copy: bool=True
    ):
    if not mol.GetNumConformers():
        conformer = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conformer)
    coords = data.pos
    if type(coords) is not np.ndarray:
        coords = coords.double().numpy()
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
    if mmff:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        except Exception as e:
            pass
    mol.n_rotable_bonds = data.edge_mask.sum()
    if not copy: return mol
    return deepcopy(mol)


class InferenceDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        for i, d in enumerate(data_list):
            d.idx = i
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def apply_torsion_and_update_pos(self, data, torsion_updates):

        pos_new, torsion_updates = perturb_batch(data, torsion_updates, split=True, return_updates=True)
        for i, idx in enumerate(data.idx):
            try:
                self.data[idx].total_perturb += torsion_updates[i]
            except:
                pass
            self.data[idx].pos = pos_new[i]

def sample_confs(
    n_confs: int,
    smi: str,
    param: dict,
    model: TensorProductScoreModel,
    pdb_dir: str | None
    ) -> list[Chem.rdchem.Mol] | None:
    seed_confs = None
    if param['seed_confs']:
        with open(param['seed_confs'], 'rb') as f:
            seed_confs = pickle.load(f)

    mol, data = get_seed(smi, param['atom_type'])
    if not mol:
        print('Failed to get seed', smi)
        return None

    n_rotable_bonds = int(data.edge_mask.sum())
    conformers, pdb = embed_seeds(
        mol, data, n_confs, single_conf=param['single_conf'], smi=smi, seed_confs=seed_confs,
        apply_pdb=param['dump_pymol'], mmff=param['pre_mmff']
        )
    if not conformers:
        print("Failed to embed", smi)
        return None

    if n_rotable_bonds > 0.5:
        conformers = perturb_seeds(conformers, pdb)

    if n_rotable_bonds > 0.5:
        conformers = sample(
            conformers, model, param['sigma_max'], param['sigma_min'], param['inference_steps'],
            param['batch_size'], param['ode'], pdb
            )

    if param['dump_pymol']:
        pdb.write(f'{pdb_dir}/{smi}.pdb', limit_parts=5)

    mols = [pyg_to_mol(mol, conf, param['post_mmff']) for conf in conformers]

    return mols
