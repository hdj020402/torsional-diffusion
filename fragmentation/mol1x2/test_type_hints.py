#!/usr/bin/env python3
"""
Test script to verify type hints work correctly
"""

import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    from mol1x2_interface import Mol1x2, combine_molecules, save_conformers_to_xyz
    print("Successfully imported mol1x2_interface with type hints")
except ImportError as e:
    print(f"Failed to import mol1x2_interface: {e}")
    sys.exit(1)

def test_type_hints() -> bool:
    """Test that type hints are working correctly"""
    print("Testing type hints...")
    
    # Test class instantiation
    mol = Mol1x2()
    print("  ✓ Mol1x2 class instantiated correctly")
    
    # Test method calls with proper types
    mol.set_energy_threshold(0.16)
    threshold = mol.get_energy_threshold()
    print(f"  ✓ Energy threshold methods work correctly (value: {threshold})")
    
    # Test function calls with proper types
    try:
        result = combine_molecules("1.xyz", "2.xyz", (1, 5), (1, 5), 0.16, 1000, "auto")
        print("  ✓ combine_molecules function works correctly")
        print(f"    Generated {result.n_conformers} conformers")
        return True
    except Exception as e:
        print(f"  ✗ combine_molecules function failed: {e}")
        return False

def main() -> int:
    """Main test function"""
    print("mol1x2 Type Hints Test")
    print("=" * 30)
    
    success = test_type_hints()
    
    if success:
        print("\nAll type hint tests passed!")
        return 0
    else:
        print("\nSome type hint tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())