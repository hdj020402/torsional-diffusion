"""
mol1x2 - Python package for combining molecular structures
"""

from typing import Tuple, Union, Any
import mol1x2 as _mol1x2
import os


class Mol1x2:
    def __init__(self) -> None:
        """Initialize the mol1x2 package"""
        _mol1x2.init_element_table()
        
    def set_energy_threshold(self, threshold: float) -> None:
        """Set the energy threshold for filtering conformers
        
        Args:
            threshold (float): Energy threshold value
        """
        _mol1x2.set_global_thresholds(threshold)
        
    def get_energy_threshold(self) -> float:
        """Get the current energy threshold
        
        Returns:
            float: Current energy threshold value
        """
        return _mol1x2.get_global_threshold_e0()
        
    def combine_structures(
        self, 
        input1: str, 
        input2: str, 
        bond1: Tuple[int, int] = (1, 5), 
        bond2: Tuple[int, int] = (1, 5), 
        max_output: int = 1000, 
        input_type: str = "auto"
    ) -> Any:
        """Combine two molecular structures from XYZ files or strings
        
        Args:
            input1 (str): Path to the first XYZ file or XYZ content string
            input2 (str): Path to the second XYZ file or XYZ content string
            bond1 (Tuple[int, int]): Bond atoms for the first molecule (1-based indices)
            bond2 (Tuple[int, int]): Bond atoms for the second molecule (1-based indices)
            max_output (int): Maximum number of output conformers
            input_type (str): Type of input - "file", "string", or "auto" (default: "auto")
            
        Returns:
            CombinedResult: Result containing combined conformers
            
        Raises:
            RuntimeError: If failed to read input files or parse input strings
        """
        # Read the first input
        mol1 = _mol1x2.XYZSet()
        if input_type == "file" or (input_type == "auto" and os.path.isfile(input1)):
            if not _mol1x2.read_xyz_all(input1, mol1):
                raise RuntimeError(f"Failed to read {input1}")
        else:
            # Assume it's a string
            result, error_msg = _mol1x2.read_xyz_from_string_with_validation(input1, mol1)
            if not result:
                raise RuntimeError(f"Failed to parse first input as XYZ string: {error_msg}")
            
        # Read the second input
        mol2 = _mol1x2.XYZSet()
        if input_type == "file" or (input_type == "auto" and os.path.isfile(input2)):
            if not _mol1x2.read_xyz_all(input2, mol2):
                raise RuntimeError(f"Failed to read {input2}")
        else:
            # Assume it's a string
            result, error_msg = _mol1x2.read_xyz_from_string_with_validation(input2, mol2)
            if not result:
                raise RuntimeError(f"Failed to parse second input as XYZ string: {error_msg}")
            
        # Combine structures
        result = _mol1x2.rpip_1x2m(
            mol1, mol2,
            bond1[0], bond1[1],
            bond2[0], bond2[1],
            max_output
        )
        
        return result


def combine_molecules(
    input1: str, 
    input2: str, 
    bond1: Tuple[int, int] = (1, 5), 
    bond2: Tuple[int, int] = (1, 5), 
    threshold: float = 0.16, 
    max_output: int = 1000, 
    input_type: str = "auto"
) -> Any:
    """Convenience function to combine two molecular structures
    
    Args:
        input1 (str): Path to the first XYZ file or XYZ content string
        input2 (str): Path to the second XYZ file or XYZ content string
        bond1 (Tuple[int, int]): Bond atoms for the first molecule (1-based indices)
        bond2 (Tuple[int, int]): Bond atoms for the second molecule (1-based indices)
        threshold (float): Energy threshold for filtering conformers
        max_output (int): Maximum number of output conformers
        input_type (str): Type of input - "file", "string", or "auto" (default: "auto")
        
    Returns:
        CombinedResult: Result containing combined conformers
        
    Raises:
        RuntimeError: If failed to read input files or parse input strings
    """
    mol = Mol1x2()
    mol.set_energy_threshold(threshold)
    return mol.combine_structures(input1, input2, bond1, bond2, max_output, input_type)


def save_conformers_to_xyz(result: Any, filename: str) -> None:
    """Save conformers to XYZ format file
    
    Args:
        result (CombinedResult): The result from combine_molecules or combine_structures
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        for i, conformer in enumerate(result.conformers):
            # Write number of atoms
            f.write(f"{conformer.nn_atom}\n")
            # Write energy line
            f.write(f"{conformer.energy_line}\n")
            # Write atom coordinates
            for j in range(conformer.nn_atom):
                # Get element symbol from atomic number
                zi = conformer.zi[j]
                # Simple mapping for common elements
                element_symbols = {
                    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
                    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"
                }
                element = element_symbols.get(zi, f"X{zi}")
                x = conformer.zb[3*j]
                y = conformer.zb[3*j+1]
                z = conformer.zb[3*j+2]
                f.write(f"{element:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


# Export classes and functions
XYZSet = _mol1x2.XYZSet
CombinedResult = _mol1x2.CombinedResult
CombinedConformer = _mol1x2.CombinedConformer