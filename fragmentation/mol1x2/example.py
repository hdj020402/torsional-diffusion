#!/usr/bin/env python3
"""
Example script showing how to use the mol1x2 package
"""

import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import mol1x2_interface as mol1x2

def main():
    print("mol1x2 Example Usage")
    print("=" * 20)
    
    # 使用便利函数组合分子
    print("Combining molecules using convenience function...")
    result = mol1x2.combine_molecules(
        "1.xyz",
        "2.xyz",
        bond1=(1, 5),
        bond2=(1, 5),
        threshold=0.16,
        max_output=100
    )
    
    print(f"Generated {result.n_conformers} conformers")
    
    # 显示前几个构象的信息
    for i, conformer in enumerate(result.conformers[:3]):
        print(f"\nConformer {i+1}:")
        print(f"  Total energy: {conformer.total_energy:.6f} a.u.")
        print(f"  Rotation angle: {conformer.rot_angle} degrees")
        print(f"  Number of atoms: {conformer.nn_atom}")
        print(f"  Energy line: {conformer.energy_line}")
        
    # 保存构象到XYZ文件
    print("\nSaving conformers to output.xyz...")
    mol1x2.save_conformers_to_xyz(result, "output.xyz")
    print("Conformers saved to output.xyz")
    
    # 使用面向对象接口
    print("\nUsing object-oriented interface...")
    combiner = mol1x2.Mol1x2()
    combiner.set_energy_threshold(0.20)  # 设置不同的阈值
    
    result2 = combiner.combine_structures(
        "1.xyz",
        "2.xyz",
        bond1=(1, 5),
        bond2=(1, 5),
        max_output=50
    )
    
    print(f"Generated {result2.n_conformers} conformers with higher threshold")
    
    # 访问单个构象的详细信息
    if result2.n_conformers > 0:
        print("\nDetailed information for first conformer:")
        first_conformer = result2.conformers[0]
        print(f"  Total energy: {first_conformer.total_energy}")
        print(f"  Energy components: E1={first_conformer.energy1}, E2={first_conformer.energy2}, E_int={first_conformer.interaction_energy}")
        print(f"  Rotation angle: {first_conformer.rot_angle}°")
        print(f"  Source conformer indices: s1={first_conformer.s1_index}, s2={first_conformer.s2_index}")
        print(f"  Number of atoms: {first_conformer.nn_atom}")
        
        # 打印前几个原子的坐标
        print("  First 3 atoms:")
        for i in range(min(3, first_conformer.nn_atom)):
            zi = first_conformer.zi[i]
            x = first_conformer.zb[3*i]
            y = first_conformer.zb[3*i+1]
            z = first_conformer.zb[3*i+2]
            # 简单的元素符号映射
            element_symbols = {
                1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
                15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"
            }
            element = element_symbols.get(zi, f"X{zi}")
            print(f"    {element}: ({x:.4f}, {y:.4f}, {z:.4f})")

if __name__ == "__main__":
    main()