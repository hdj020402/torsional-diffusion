#!/usr/bin/env python3
"""
Example script showing how to use the mol1x2 package with string input
"""

import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import mol1x2_interface as mol1x2

def main():
    print("mol1x2 String Input Example")
    print("=" * 30)
    
    # 方法1：使用文件路径（原有方式）
    print("Method 1: Using file paths")
    result1 = mol1x2.combine_molecules("1.xyz", "2.xyz")
    print(f"Generated {result1.n_conformers} conformers from files")
    
    # 方法2：使用字符串（新增方式）
    print("\nMethod 2: Using string content")
    with open("1.xyz", "r") as f:
        xyz1_content = f.read()
    
    with open("2.xyz", "r") as f:
        xyz2_content = f.read()
    
    result2 = mol1x2.combine_molecules(xyz1_content, xyz2_content, input_type="string")
    print(f"Generated {result2.n_conformers} conformers from strings")
    
    # 方法3：混合使用（新增方式）
    print("\nMethod 3: Mixed usage (file and string)")
    result3 = mol1x2.combine_molecules("1.xyz", xyz2_content, input_type="auto")
    print(f"Generated {result3.n_conformers} conformers from mixed input")
    
    # 显示结果的一致性
    print("\nResult consistency check:")
    print(f"  File input result: {result1.n_conformers} conformers")
    print(f"  String input result: {result2.n_conformers} conformers")
    print(f"  Mixed input result: {result3.n_conformers} conformers")
    
    if result1.n_conformers == result2.n_conformers == result3.n_conformers:
        print("  All methods produced consistent results!")
    else:
        print("  WARNING: Inconsistent results between methods!")
    
    # 显示第一个构象的详细信息
    if result2.n_conformers > 0:
        print("\nDetailed information for first conformer (from string input):")
        first_conformer = result2.conformers[0]
        print(f"  Total energy: {first_conformer.total_energy:.6f} a.u.")
        print(f"  Energy components: E1={first_conformer.energy1:.6f}, E2={first_conformer.energy2:.6f}, E_int={first_conformer.interaction_energy:.6f}")
        print(f"  Rotation angle: {first_conformer.rot_angle}°")
        print(f"  Source conformer indices: s1={first_conformer.s1_index}, s2={first_conformer.s2_index}")
        print(f"  Number of atoms: {first_conformer.nn_atom}")
        
        # 打印前几个原子的坐标
        print("  First 3 atoms:")
        for i in range(min(3, first_conformer.nn_atom)):  # Print first 3 atoms
            zi = first_conformer.zi[i]  # Atomic number
            x = first_conformer.zb[3*i]    # X coordinate
            y = first_conformer.zb[3*i+1]  # Y coordinate
            z = first_conformer.zb[3*i+2]  # Z coordinate
            # 简单的元素符号映射
            element_symbols = {
                1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
                15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"
            }
            element = element_symbols.get(zi, f"X{zi}")
            print(f"    {element}: ({x:.4f}, {y:.4f}, {z:.4f})")
    
    # 保存构象到XYZ文件
    print("\nSaving conformers to output_from_strings.xyz...")
    mol1x2.save_conformers_to_xyz(result2, "output_from_strings.xyz")
    print("Conformers saved to output_from_strings.xyz")

if __name__ == "__main__":
    main()