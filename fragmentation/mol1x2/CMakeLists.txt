cmake_minimum_required(VERSION 3.12)
project(mol1x2)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

# 查找pybind11
find_package(pybind11 REQUIRED)

# 包含目录
include_directories(include)

# 创建共享库
add_library(mol1x2_wrapper SHARED src/mol1x2_wrapper.c)
target_compile_definitions(mol1x2_wrapper PRIVATE PYBIND11_COMPILER_TYPE="_gcc" PYBIND11_STDLIB="_libstdcpp")

# 创建Python绑定
pybind11_add_module(mol1x2 python/mol1x2.cpp)

# 链接库
target_link_libraries(mol1x2 PRIVATE mol1x2_wrapper)

# 安装规则
install(TARGETS mol1x2 DESTINATION .)