"""
mol1x2 - Python package for combining molecular structures
"""

import mol1x2 as _mol1x2

class Mol1x2:
    def __init__(self):
        """Initialize the mol1x2 package"""
        _mol1x2.init_element_table()
        
    def set_energy_threshold(self, threshold):
        """Set the energy threshold for filtering conformers
        
        Args:
            threshold (float): Energy threshold value
        """
        _mol1x2.set_global_thresholds(threshold)
        
    def get_energy_threshold(self):
        """Get the current energy threshold
        
        Returns:
            float: Current energy threshold value
        """
        return _mol1x2.get_global_threshold_e0()
        
    def combine_structures(self, file1, file2, bond1=(1, 5), bond2=(1, 5), max_output=1000):
        """Combine two molecular structures from XYZ files
        
        Args:
            file1 (str): Path to the first XYZ file
            file2 (str): Path to the second XYZ file
            bond1 (tuple): Bond atoms for the first molecule (1-based indices)
            bond2 (tuple): Bond atoms for the second molecule (1-based indices)
            max_output (int): Maximum number of output conformers
            
        Returns:
            CombinedResult: Result containing combined conformers
        """
        # Read the first XYZ file
        mol1 = _mol1x2.XYZSet()
        if not _mol1x2.read_xyz_all(file1, mol1):
            raise RuntimeError(f"Failed to read {file1}")
            
        # Read the second XYZ file
        mol2 = _mol1x2.XYZSet()
        if not _mol1x2.read_xyz_all(file2, mol2):
            raise RuntimeError(f"Failed to read {file2}")
            
        # Combine structures
        result = _mol1x2.rpip_1x2m(
            mol1, mol2,
            bond1[0], bond1[1],
            bond2[0], bond2[1],
            max_output
        )
        
        return result

# Convenience function
def combine_molecules(file1, file2, bond1=(1, 5), bond2=(1, 5), threshold=0.16, max_output=1000):
    """Convenience function to combine two molecular structures
    
    Args:
        file1 (str): Path to the first XYZ file
        file2 (str): Path to the second XYZ file
        bond1 (tuple): Bond atoms for the first molecule (1-based indices)
        bond2 (tuple): Bond atoms for the second molecule (1-based indices)
        threshold (float): Energy threshold for filtering conformers
        max_output (int): Maximum number of output conformers
        
    Returns:
        CombinedResult: Result containing combined conformers
    """
    mol = Mol1x2()
    mol.set_energy_threshold(threshold)
    return mol.combine_structures(file1, file2, bond1, bond2, max_output)