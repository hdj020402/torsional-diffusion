from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

ext_modules = [
    Pybind11Extension(
        "mol1x2",
        [
            "python/mol1x2.cpp",
            "src/mol1x2_wrapper.c"
        ],
        include_dirs=[
            "include",
            pybind11.get_include()
        ],
        cxx_std=11,
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    name="mol1x2",
    version="0.1.0",
    author="mol1x2",
    description="Python bindings for mol1x2 molecular structure combination tool",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)