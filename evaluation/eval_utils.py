import pickle, random, pandas as pd
import numpy as np
from typing import Callable
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem

RMSD_THRESHOLDS = np.arange(0, 2.5, .125)

@dataclass
class EvaluationMetrics:
    RMSD_THRESHOLDS: np.ndarray
    num_model_failures: int
    num_additional_failures: int
    all_recall_coverages: np.ndarray
    all_amr_recalls: np.ndarray
    all_precision_coverages: np.ndarray
    all_amr_precisions: np.ndarray
    evaluation_results: dict

# --- Helper Function: Calculate Performance Statistics ---
def calculate_performance_stats(rmsd_matrix: np.ndarray) -> tuple:
    """
    Calculates performance statistics for a conformer generation model based on an RMSD matrix.

    Args:
        rmsd_matrix (np.ndarray): A 2D NumPy array where rm_matrix[i, j] represents the RMSD
                                between the i-th true conformer and the j-th model conformer.
                                Shape: (num_true_conformers, num_model_conformers).

    Returns:
        tuple: A tuple containing the following four statistical metrics:
            - coverage_recall (np.ndarray): Recall coverage for each threshold.
            - amr_recall (float): Average Minimum RMSD Recall.
            - coverage_precision (np.ndarray): Precision coverage for each threshold.
            - amr_precision (float): Average Minimum RMSD Precision.
    """
    # Recall Coverage:
    # Measures the proportion of true conformers for which at least one model conformer
    # is within a given RMSD threshold.
    # rmsd_matrix.min(axis=1, keepdims=True) gets the minimum RMSD for each true conformer.
    # < RMSD_THRESHOLDS performs element-wise comparison.
    # np.mean(..., axis=0) averages along the threshold axis.
    coverage_recall = np.mean(rmsd_matrix.min(axis=1, keepdims=True) < RMSD_THRESHOLDS, axis=0)

    # Average Minimum RMSD Recall (AMR Recall):
    # The average of the minimum RMSDs between all true conformers and their closest model conformer.
    amr_recall = rmsd_matrix.min(axis=1).mean()

    # Precision Coverage:
    # Measures the proportion of model conformers for which at least one true conformer
    # is within a given RMSD threshold.
    # rmsd_matrix.min(axis=0, keepdims=True) gets the minimum RMSD for each model conformer.
    # np.expand_dims(RMSD_THRESHOLDS, 1) expands dimensions for broadcasting comparison.
    # np.mean(..., axis=1) averages along the conformer axis.
    coverage_precision = np.mean(rmsd_matrix.min(axis=0, keepdims=True) < np.expand_dims(RMSD_THRESHOLDS, 1), axis=1)

    # Average Minimum RMSD Precision (AMR Precision):
    # The average of the minimum RMSDs between all model conformers and their closest true conformer.
    amr_precision = rmsd_matrix.min(axis=0).mean()

    return coverage_recall, amr_recall, coverage_precision, amr_precision

# --- Function to Aggregate Evaluation Results ---
def aggregate_evaluation_results(
    evaluation_results: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Aggregates molecule-level evaluation results and handles additional failures.

    Args:
        evaluation_results (dict): Dictionary containing per-molecule evaluation results,
                                   including the 'rmsd' matrix.

    Returns:
        tuple: A tuple containing:
               - all_recall_coverages (np.ndarray): Aggregated recall coverages across all molecules.
               - all_amr_recalls (list): Aggregated AMR recalls across all molecules.
               - all_precision_coverages (np.ndarray): Aggregated precision coverages across all molecules.
               - all_amr_precisions (list): Aggregated AMR precisions across all molecules.
               - num_additional_failures (int): Count of molecules with additional failures.
    """
    all_recall_coverages = []
    all_amr_recalls = []
    all_precision_coverages = []
    all_amr_precisions = []

    num_additional_failures = 0

    # Iterate through the evaluation results for each molecule
    for mol_data in evaluation_results.values():
        current_rmsd_matrix: np.ndarray = mol_data['rmsd']

        # Check if the RMSD matrix contains any NaN values (indicating additional failures)
        if np.isnan(current_rmsd_matrix).any():
            num_additional_failures += 1
            # Set the statistics for this molecule to NaN.
            # These NaNs will be ignored by np.nanmean/nanmedian later.
            stats = (np.nan * np.ones_like(RMSD_THRESHOLDS), np.nan,
                     np.nan * np.ones_like(RMSD_THRESHOLDS), np.nan)
        else:
            # Calculate statistics for the molecule if no failures occurred in RMSD calculation
            stats = calculate_performance_stats(current_rmsd_matrix)

        # Append the statistics (either calculated or NaN) to the lists
        all_recall_coverages.append(stats[0])
        all_amr_recalls.append(stats[1])
        all_precision_coverages.append(stats[2])
        all_amr_precisions.append(stats[3])

    # Convert collected lists to NumPy arrays for easier statistical calculations
    # Note: all_amr_recalls and all_amr_precisions are lists of floats/NaNs,
    # they can be converted to numpy arrays if needed for further operations,
    # but are sufficient as lists for nanmean/nanmedian directly.
    # However, converting consistently is good practice.
    all_recall_coverages = np.array(all_recall_coverages)
    all_amr_recalls = np.array(all_amr_recalls) # Convert to numpy array
    all_precision_coverages = np.array(all_precision_coverages)
    all_amr_precisions = np.array(all_amr_precisions) # Convert to numpy array


    return (all_recall_coverages, all_amr_recalls,
            all_precision_coverages, all_amr_precisions,
            num_additional_failures)

# --- Helper Function: Clean Conformers ---
def clean_conformers(smiles_str: str, conformers: list) -> list:
    """
    Cleans and filters a list of RDKit conformer objects based on a SMILES string.
    Only conformers matching the topological structure of the input SMILES string are retained.

    Args:
        smiles_str (str): The reference SMILES string used to validate conformers.
        conformers (list): A list of RDKit molecule conformer objects.

    Returns:
        list: A filtered list of RDKit molecule conformer objects.
    """
    cleaned_conformers = []
    # Normalize the input SMILES string
    try:
        normalized_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_str, sanitize=False), isomericSmiles=False)
    except Exception as e:
        # print(f"Warning: Could not normalize SMILES '{smiles_str}': {e}. Skipping this molecule's cleaning.")
        return []

    for conf in conformers:
        try:
            # Remove hydrogen atoms and normalize the conformer's SMILES for comparison
            conf_smi = Chem.MolToSmiles(Chem.RemoveHs(conf, sanitize=False), isomericSmiles=False)
            if conf_smi == normalized_smi:
                cleaned_conformers.append(conf)
        except Exception as e:
            # print(f"Warning: Could not process conformer during cleaning for SMILES '{smiles_str}': {e}")
            continue # Skip problematic conformers

    return cleaned_conformers

def preparation(param: dict, trace_func: Callable=print) -> tuple[int, dict[str, dict], list]:
    with open(param['true_confs'], 'rb') as f:
        true_molecules: dict = pickle.load(f)
    with open(param['gen_confs'], 'rb') as f:
        model_predictions: dict = pickle.load(f)

    test_data = pd.read_csv(param['test_csv'])

    num_model_failures = 0
    evaluation_results: dict[str, dict] = {}
    rmsds_calculation_jobs = []

    for smi in test_data['smiles']:

        if model_predictions[smi] is None:
            trace_func(f'Model failed to generate conformers for molecule {smi}. Skipping evaluation.')
            num_model_failures += 1
            continue

        true_molecules[smi] = cleaned_true_confs = clean_conformers(smi, true_molecules.get(smi, []))

        if len(cleaned_true_confs) == 0:
            trace_func(f'Molecule {smi} has 0 true conformers. Cannot evaluate.')
            continue

        num_true_confs = len(cleaned_true_confs)
        num_model_confs = len(model_predictions[smi])

        evaluation_results[smi] = {
            'num_true_confs': num_true_confs,
            'num_model_confs': num_model_confs,
            'rmsd': np.nan * np.ones((num_true_confs, num_model_confs))
        }

        for i_true in range(num_true_confs):
            rmsds_calculation_jobs.append((smi, i_true, true_molecules[smi][i_true], model_predictions[smi]))

    random.shuffle(rmsds_calculation_jobs)

    return num_model_failures, evaluation_results, rmsds_calculation_jobs

# --- Worker Function: Calculate RMSDs for a Single True Conformer ---
def calculate_rmsds_for_true_conformer(job_data: tuple, trace_func: Callable=print) -> tuple:
    """
    A worker function to calculate RMSDs between a single true conformer
    and all model conformers for a given molecule.
    This function is intended to be called by a multiprocessing Pool.

    Args:
        job_data (tuple): A tuple containing the following information:
                        - smi (str): SMILES string.
                        - true_conf_idx (int): The index of the current true conformer
                                                within `true_molecules_map[mol_smi]`.
                        - true_molecules_map (dict): A dictionary containing all true conformers.
                        - model_predictions_map (dict): A dictionary containing all model predicted conformers.

    Returns:
        tuple: A tuple containing SMILES,
            true conformer index, and a list of corresponding RMSDs.
            The RMSD list will be filled with NaN if a calculation fails.
    """
    smi, true_conf_idx, current_true_conf, model_conformers = job_data

    rmsds = []
    for model_conf in model_conformers:
        try:
            # Remove hydrogen atoms before calculating RMSD
            true_mol_no_hs = Chem.RemoveHs(current_true_conf)
            model_mol_no_hs = Chem.RemoveHs(model_conf)
            
            rmsd = AllChem.GetBestRMS(true_mol_no_hs, model_mol_no_hs)
            rmsds.append(rmsd)
        except Exception as e:
            # Record calculation failure and fill with NaN to identify issues in stats
            trace_func(f'Additional failure during RMSD calculation for {smi}, true conf index {true_conf_idx}: {e}')
            rmsds = [np.nan] * len(model_conformers)
            break # If an error occurs, all subsequent RMSDs for this true conformer are invalid

    return smi, true_conf_idx, rmsds
