# mol1x2 Python Package

Python bindings for the mol1x2 molecular structure combination tool.

## Description

This package provides Python bindings for a C program that combines two molecular structures from XYZ files. It reads two XYZ files, aligns them based on specified bonding atoms, rotates one molecule around the bond axis, and generates multiple conformers with different rotation angles.

## Features

- Read XYZ files containing molecular structures
- Read XYZ data from strings in memory
- Combine two molecular structures at specified bonding atoms
- Generate multiple conformers with different rotation angles
- Filter conformers based on energy thresholds
- Access detailed information about each conformer in memory
- Save conformers to XYZ format files
- Support mixed input (one file, one string)

## Installation

### Prerequisites

- Python 3.6 or higher
- pybind11
- A C/C++ compiler (gcc/clang)

### Installation with pip

```bash
pip install pybind11
cd mol1x2
pip install .
```

### Installation with CMake

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Basic Usage

```python
import mol1x2_interface as mol1x2

# Use the convenience function for simple cases
result = mol1x2.combine_molecules(
    "molecule1.xyz",
    "molecule2.xyz",
    bond1=(1, 5),      # Bond atoms for first molecule (1-based indices)
    bond2=(1, 5),      # Bond atoms for second molecule (1-based indices)
    threshold=0.16,    # Energy threshold for filtering conformers
    max_output=1000    # Maximum number of output conformers
)

# Access results in memory
print(f"Generated {result.n_conformers} conformers")
for i, conformer in enumerate(result.conformers):
    print(f"Conformer {i}: Energy = {conformer.total_energy}")

# Save conformers to XYZ file
mol1x2.save_conformers_to_xyz(result, "output.xyz")
```

### String Input Usage

```python
import mol1x2_interface as mol1x2

# Read XYZ content from files or other sources
with open("molecule1.xyz", "r") as f:
    xyz1_content = f.read()

with open("molecule2.xyz", "r") as f:
    xyz2_content = f.read()

# Combine molecules from string content
result = mol1x2.combine_molecules(
    xyz1_content,
    xyz2_content,
    bond1=(1, 5),
    bond2=(1, 5),
    threshold=0.16,
    max_output=1000,
    input_type="string"  # Specify that inputs are strings
)

# Save results
mol1x2.save_conformers_to_xyz(result, "output_from_strings.xyz")
```

### Mixed Input Usage

```python
import mol1x2_interface as mol1x2

# Read one molecule from file, another from string
with open("molecule2.xyz", "r") as f:
    xyz2_content = f.read()

# Combine with mixed input (auto-detection)
result = mol1x2.combine_molecules(
    "molecule1.xyz",     # File path
    xyz2_content,        # String content
    bond1=(1, 5),
    bond2=(1, 5),
    threshold=0.16,
    max_output=1000,
    input_type="auto"    # Auto-detect input types
)
```

### Object-Oriented Interface

```python
import mol1x2_interface as mol1x2

# Create a combiner instance
combiner = mol1x2.Mol1x2()

# Set energy threshold (optional)
combiner.set_energy_threshold(0.16)

# Combine two structures
result = combiner.combine_structures(
    "molecule1.xyz",   # First XYZ file or string
    "molecule2.xyz",   # Second XYZ file or string
    bond1=(1, 5),      # Bond atoms for first molecule (1-based indices)
    bond2=(1, 5),      # Bond atoms for second molecule (1-based indices)
    max_output=1000,   # Maximum number of output conformers
    input_type="auto"  # Auto-detect input types
)

# Access detailed information for each conformer
if result.n_conformers > 0:
    first_conformer = result.conformers[0]
    print(f"Total energy: {first_conformer.total_energy}")
    print(f"Energy components: E1={first_conformer.energy1}, E2={first_conformer.energy2}, E_int={first_conformer.interaction_energy}")
    print(f"Rotation angle: {first_conformer.rot_angle}°")
    print(f"Source conformer indices: s1={first_conformer.s1_index}, s2={first_conformer.s2_index}")
    
    # Access atomic information
    print(f"Number of atoms: {first_conformer.nn_atom}")
    for i in range(min(3, first_conformer.nn_atom)):  # Print first 3 atoms
        zi = first_conformer.zi[i]  # Atomic number
        x = first_conformer.zb[3*i]    # X coordinate
        y = first_conformer.zb[3*i+1]  # Y coordinate
        z = first_conformer.zb[3*i+2]  # Z coordinate
        print(f"  Atom {i}: Z={zi}, ({x:.4f}, {y:.4f}, {z:.4f})")
```

## API Reference

### Mol1x2 Class

#### `__init__()`
Initialize the mol1x2 package.

#### `set_energy_threshold(threshold)`
Set the energy threshold for filtering conformers.

#### `get_energy_threshold()`
Get the current energy threshold.

#### `combine_structures(input1, input2, bond1, bond2, max_output, input_type)`
Combine two molecular structures from XYZ files or strings.

### Convenience Functions

#### `combine_molecules(input1, input2, bond1, bond2, threshold, max_output, input_type)`
Convenience function to combine two molecular structures.

#### `save_conformers_to_xyz(result, filename)`
Save conformers to XYZ format file.

## Data Structures

### CombinedResult
- `n_conformers`: Number of generated conformers
- `conformers`: List of CombinedConformer objects

### CombinedConformer
- `total_energy`: Total energy of the conformer (atomic units)
- `energy1`: Energy of the first molecule
- `energy2`: Energy of the second molecule
- `interaction_energy`: Interaction energy between molecules
- `rot_angle`: Rotation angle (degrees)
- `s1_index`: Index of the first molecule conformer
- `s2_index`: Index of the second molecule conformer
- `nn_atom`: Total number of atoms
- `zi`: List of atomic numbers
- `zb`: List of atomic coordinates (x, y, z triplets)
- `energy_line`: Energy information line

## Adjustable Parameters

1. Input XYZ file paths or string content
2. Bonding atom indices (bond1 and bond2)
3. Energy threshold (via `set_energy_threshold` function)
4. Maximum number of output conformers (max_output)
5. Input type (file, string, or auto-detection)

## Performance

Performance testing shows that reading XYZ data from strings in memory is approximately 136 times faster than reading directly from files, making it the preferred method when XYZ data is already available in memory.

## Testing

Run the test scripts to verify the installation:

```bash
python test.py           # Test file-based input
python test_string_input.py  # Test string-based input
python performance_test.py   # Performance comparison
```

## Example

See `example.py` for file-based usage and `example_string_input.py` for string-based usage.

## License

This project is licensed under the MIT License.