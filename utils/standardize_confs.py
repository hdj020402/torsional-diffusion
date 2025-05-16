import glob, os, pickle, random, tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from utils.standardization import *

RDLogger.DisableLog('rdApp.*')

REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)


def sort_confs(confs):
    return sorted(confs, key=lambda conf: -conf['boltzmannweight'])


def resample_confs(confs, max_confs=None):
    weights = [conf['boltzmannweight'] for conf in confs]
    weights = np.array(weights) / sum(weights)
    k = min(max_confs, len(confs)) if max_confs else len(confs)
    return random.choices(confs, weights, k=k)


def log_error(err):
    print(err)
    long_term_log[err] += 1
    return None


def conformer_match(name, confs):
    long_term_log['confs_seen'] += len(confs)

    if args.boltzmann == 'top':
        confs = sort_confs(confs)

    limit = args.confs_per_mol if args.boltzmann != 'resample' else None
    confs = clean_confs(name, confs, limit=limit)
    if not confs: return log_error("no_clean_confs")

    if args.boltzmann == 'resample':
        confs = resample_confs(confs, max_confs=args.confs_per_mol)

    if args.confs_per_mol:
        confs = confs[:args.confs_per_mol]

    n_confs = len(confs)

    new_confs = []

    mol_rdkit = copy.deepcopy(confs[0]['rd_mol'])
    rotable_bonds = get_torsion_angles(mol_rdkit)

    if not rotable_bonds: return log_error("no_rotable_bonds")

    mol_rdkit.RemoveAllConformers()
    AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=n_confs)

    if mol_rdkit.GetNumConformers() != n_confs:
        return log_error("rdkit_no_embed")
    if args.mmff:
        try:
            mmff_func(mol_rdkit)
        except:
            return log_error("mmff_error")

    if not args.no_match:
        cost_matrix = [[get_von_mises_rms(confs[i]['rd_mol'], mol_rdkit, rotable_bonds, j) for j in range(n_confs)] for
                       i in range(n_confs)]
        cost_matrix = np.asarray(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        row_ind, col_ind = np.arange(len(confs)), np.arange(len(confs))

    iterable = tqdm.tqdm(enumerate(confs), total=len(confs))

    for i, conf in iterable:
        mol = conf['rd_mol']
        conf_id = int(col_ind[i])

        try:
            mol_rdkit_single = copy.deepcopy(mol_rdkit)
            [mol_rdkit_single.RemoveConformer(j) for j in range(n_confs) if j != conf_id]
            optimize_rotatable_bonds(mol_rdkit_single, mol, rotable_bonds,
                                     popsize=args.popsize, maxiter=args.max_iter)
            rmsd = AllChem.AlignMol(REMOVE_HS(mol_rdkit_single), REMOVE_HS(mol))
            long_term_log['confs_success'] += 1

        except Exception as e:
            print(e)
            long_term_log['confs_fail'] += 1
            continue

        conf['rd_mol'] = mol_rdkit_single
        conf['rmsd'] = rmsd
        conf['num_rotable_bonds'] = len(rotable_bonds)
        new_confs.append(conf)

        long_term_rmsd_cache.append(rmsd)
    return new_confs


with open(true_conformers, 'rb') as tc:
    db: dict = pickle.load(tc)
for key, mol_dict in db.items():
    confs = mol_dict['conformers']
    smiles = mol_dict["smiles"]

    try:
        new_confs = conformer_match(smiles, confs)
    except Exception as e:
        print(e)
        long_term_log['mol_other_failure'] += 1
        new_confs = None

    if not new_confs:
        print(f'{i} Failure nconfs={len(confs)} smi={name}')
    else:
        num_rotable_bonds = new_confs[0]['num_rotable_bonds']
        rmsds = [conf['rmsd'] for conf in new_confs]
        print(
            f'{i} Success nconfs={len(new_confs)}/{len(confs)} bonds={num_rotable_bonds} rmsd={np.mean(rmsds):.2f} smi={name}')
        mol_dic['conformers'] = new_confs
        master_dict[f[len(root):-7]] = mol_dic

    if (i + 1) % 20 == 0:
        update = {
             'mols_processed': i + 1,
             'mols_success': len(master_dict),
             'mean_rmsd': np.mean(long_term_rmsd_cache)
        } | long_term_log
        print(update)
print('ALL RMSD', np.mean(long_term_rmsd_cache))
if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
with open(args.out_dir + '/' + str(args.worker_id).zfill(3) + '.pickle', 'wb') as f:
    pickle.dump(master_dict, f)

