#!/usr/bin/env python3
"""
Performance test script for mol1x2 Python package
Compares file-based input vs string-based input
"""

import sys
import os
import time

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    import mol1x2
    print("Successfully imported mol1x2")
except ImportError as e:
    print(f"Failed to import mol1x2: {e}")
    sys.exit(1)

def test_file_based_approach():
    """Test the file-based approach"""
    # 初始化元素表
    mol1x2.init_element_table()
    
    # 设置能量阈值
    mol1x2.set_global_thresholds(0.16)
    
    start_time = time.time()
    
    # 读取XYZ文件
    mol1 = mol1x2.XYZSet()
    mol2 = mol1x2.XYZSet()
    
    if not mol1x2.read_xyz_all("1.xyz", mol1):
        print("Failed to read first molecule from file")
        return None
        
    if not mol1x2.read_xyz_all("2.xyz", mol2):
        print("Failed to read second molecule from file")
        return None
    
    # 组合分子
    result = mol1x2.rpip_1x2m(
        mol1, mol2,
        1, 5,  # bond1
        1, 5,  # bond2
        100    # max_output
    )
    
    end_time = time.time()
    
    return end_time - start_time, result.n_conformers

def test_string_based_approach():
    """Test the string-based approach"""
    # 读取文件内容到内存
    with open("1.xyz", "r") as f:
        xyz1_content = f.read()
    
    with open("2.xyz", "r") as f:
        xyz2_content = f.read()
    
    # 初始化元素表
    mol1x2.init_element_table()
    
    # 设置能量阈值
    mol1x2.set_global_thresholds(0.16)
    
    start_time = time.time()
    
    # 从字符串读取分子数据
    mol1 = mol1x2.XYZSet()
    mol2 = mol1x2.XYZSet()
    
    if not mol1x2.read_xyz_from_string(xyz1_content, mol1):
        print("Failed to read first molecule from string")
        return None
        
    if not mol1x2.read_xyz_from_string(xyz2_content, mol2):
        print("Failed to read second molecule from string")
        return None
    
    # 组合分子
    result = mol1x2.rpip_1x2m(
        mol1, mol2,
        1, 5,  # bond1
        1, 5,  # bond2
        100    # max_output
    )
    
    end_time = time.time()
    
    return end_time - start_time, result.n_conformers

def run_performance_comparison():
    """Run performance comparison between file-based and string-based approaches"""
    print("Performance Comparison Test")
    print("=" * 40)
    
    # Run multiple iterations for better accuracy
    num_iterations = 10
    file_times = []
    string_times = []
    
    print(f"Running {num_iterations} iterations for each approach...")
    
    # Test file-based approach
    print("Testing file-based approach...")
    for i in range(num_iterations):
        time_taken, conformers = test_file_based_approach()
        if time_taken is not None:
            file_times.append(time_taken)
            if i == 0:  # Print conformer count only once
                print(f"  Generated {conformers} conformers")
    
    # Test string-based approach
    print("Testing string-based approach...")
    for i in range(num_iterations):
        time_taken, conformers = test_string_based_approach()
        if time_taken is not None:
            string_times.append(time_taken)
            if i == 0:  # Print conformer count only once
                print(f"  Generated {conformers} conformers")
    
    # Calculate statistics
    if file_times and string_times:
        avg_file_time = sum(file_times) / len(file_times)
        avg_string_time = sum(string_times) / len(string_times)
        
        print("\nPerformance Results:")
        print(f"  File-based approach:  {avg_file_time:.6f} seconds (average)")
        print(f"  String-based approach: {avg_string_time:.6f} seconds (average)")
        
        if avg_file_time < avg_string_time:
            speedup = avg_string_time / avg_file_time
            print(f"  File-based is {speedup:.2f}x faster")
        else:
            speedup = avg_file_time / avg_string_time
            print(f"  String-based is {speedup:.2f}x faster")
        
        return True
    else:
        print("Failed to collect performance data")
        return False

def main():
    """Main function"""
    success = run_performance_comparison()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())