#!/usr/bin/env python3
"""
Test script for mol1x2 Python package - String input functionality
"""

import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    import mol1x2
    print("Successfully imported mol1x2")
except ImportError as e:
    print(f"Failed to import mol1x2: {e}")
    sys.exit(1)

def test_string_input():
    """Test reading XYZ data from strings"""
    print("Testing string input functionality...")
    
    # 读取文件内容到内存
    with open("1.xyz", "r") as f:
        xyz1_content = f.read()
    
    with open("2.xyz", "r") as f:
        xyz2_content = f.read()
    
    print("File contents loaded into memory")
    
    # 使用字符串输入方式
    mol1 = mol1x2.XYZSet()
    mol2 = mol1x2.XYZSet()
    
    # 测试基本字符串读取
    print("Testing basic string reading...")
    result1 = mol1x2.read_xyz_from_string(xyz1_content, mol1)
    result2 = mol1x2.read_xyz_from_string(xyz2_content, mol2)
    
    if result1 and result2:
        print(f"Successfully read XYZ data from strings")
        print(f"  Molecule 1: {mol1.nn_atom} atoms, {mol1.stn} conformers")
        print(f"  Molecule 2: {mol2.nn_atom} atoms, {mol2.stn} conformers")
    else:
        print("Failed to read XYZ data from strings")
        return False
    
    # 测试带验证的字符串读取
    print("Testing string reading with validation...")
    
    # 创建新的XYZSet对象用于测试
    mol1_test = mol1x2.XYZSet()
    mol2_test = mol1x2.XYZSet()
    
    result3, error_msg = mol1x2.read_xyz_from_string_with_validation(xyz1_content, mol1_test)
    result4, error_msg2 = mol1x2.read_xyz_from_string_with_validation(xyz2_content, mol2_test)
    
    if result3 and result4:
        print(f"Successfully read XYZ data from strings with validation")
    else:
        print(f"Failed to read XYZ data from strings with validation: {error_msg or error_msg2}")
        return False
    
    # 测试错误输入
    print("Testing error handling...")
    invalid_content = "invalid xyz content"
    
    # 创建新的XYZSet对象用于测试
    mol_invalid = mol1x2.XYZSet()
    result5, error_msg = mol1x2.read_xyz_from_string_with_validation(invalid_content, mol_invalid)
    if not result5:
        print(f"Correctly handled invalid input: {error_msg}")
    else:
        print("Failed to detect invalid input")
        return False
    
    # 测试空字符串
    print("Testing empty string handling...")
    empty_content = ""
    mol_empty = mol1x2.XYZSet()
    result6, error_msg = mol1x2.read_xyz_from_string_with_validation(empty_content, mol_empty)
    if not result6:
        print(f"Correctly handled empty input: {error_msg}")
    else:
        print("Failed to detect empty input")
        return False
    
    return True

def test_combine_from_strings():
    """Test combining molecules from string input"""
    print("\nTesting combination from string input...")
    
    # 读取文件内容到内存
    with open("1.xyz", "r") as f:
        xyz1_content = f.read()
    
    with open("2.xyz", "r") as f:
        xyz2_content = f.read()
    
    # 初始化元素表
    mol1x2.init_element_table()
    
    # 设置能量阈值
    mol1x2.set_global_thresholds(0.16)
    
    # 从字符串读取分子数据
    mol1 = mol1x2.XYZSet()
    mol2 = mol1x2.XYZSet()
    
    if not mol1x2.read_xyz_from_string(xyz1_content, mol1):
        print("Failed to read first molecule from string")
        return False
        
    if not mol1x2.read_xyz_from_string(xyz2_content, mol2):
        print("Failed to read second molecule from string")
        return False
    
    # 组合分子
    result = mol1x2.rpip_1x2m(
        mol1, mol2,
        1, 5,  # bond1
        1, 5,  # bond2
        100    # max_output
    )
    
    print(f"Generated {result.n_conformers} conformers from string input")
    
    if result.n_conformers > 0:
        # 显示第一个构象的信息
        first = result.conformers[0]
        print(f"First conformer:")
        print(f"  Total energy: {first.total_energy}")
        print(f"  Rotation angle: {first.rot_angle}")
        print(f"  Number of atoms: {first.nn_atom}")
        return True
    else:
        print("No conformers generated")
        return False

def main():
    """Main test function"""
    print("mol1x2 String Input Test")
    print("=" * 30)
    
    success1 = test_string_input()
    success2 = test_combine_from_strings()
    
    if success1 and success2:
        print("\nAll string input tests passed!")
        return 0
    else:
        print("\nSome string input tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())