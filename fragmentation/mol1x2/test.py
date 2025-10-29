#!/usr/bin/env python3
"""
Test script for mol1x2 Python package
"""

import os
import sys

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    import mol1x2_interface as mol1x2
    print("Successfully imported mol1x2")
except ImportError as e:
    print(f"Failed to import mol1x2: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of the mol1x2 package"""
    print("Testing basic functionality...")
    
    # Initialize the package
    combiner = mol1x2.Mol1x2()
    
    # Test setting and getting energy threshold
    combiner.set_energy_threshold(0.16)
    threshold = combiner.get_energy_threshold()
    print(f"Energy threshold: {threshold}")
    
    # Check if XYZ files exist
    if not os.path.exists("1.xyz") or not os.path.exists("2.xyz"):
        print("Error: 1.xyz or 2.xyz not found in current directory")
        return False
        
    try:
        # Combine structures
        print("Combining structures...")
        result = combiner.combine_structures(
            "1.xyz", 
            "2.xyz", 
            bond1=(1, 5), 
            bond2=(1, 5),
            max_output=100
        )
        
        print(f"Generated {result.n_conformers} conformers")
        
        if result.n_conformers > 0:
            # Print information about the first conformer
            first = result.conformers[0]
            print(f"First conformer:")
            print(f"  Total energy: {first.total_energy}")
            print(f"  Rotation angle: {first.rot_angle}")
            print(f"  Number of atoms: {first.nn_atom}")
            print(f"  Energy line: {first.energy_line}")
            return True
        else:
            print("No conformers generated")
            return False
            
    except Exception as e:
        print(f"Error during combination: {e}")
        return False

def test_convenience_function():
    """Test the convenience function"""
    print("\nTesting convenience function...")
    
    try:
        result = mol1x2.combine_molecules(
            "1.xyz",
            "2.xyz",
            bond1=(1, 5),
            bond2=(1, 5),
            threshold=0.16,
            max_output=50
        )
        
        print(f"Generated {result.n_conformers} conformers using convenience function")
        print(result.conformers[0])
        return result.n_conformers > 0
    except Exception as e:
        print(f"Error in convenience function: {e}")
        return False

def main():
    """Main test function"""
    print("mol1x2 Python Package Test")
    print("=" * 30)
    
    success1 = test_basic_functionality()
    success2 = test_convenience_function()
    
    if success1 and success2:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())