"""
Type definitions for mol1x2 Python package
"""

from typing import List, Tuple, Union
from typing_extensions import TypedDict


class CombinedConformer(TypedDict):
    """TypedDict for CombinedConformer structure"""
    total_energy: float
    energy1: float
    energy2: float
    interaction_energy: float
    rot_angle: int
    s1_index: int
    s2_index: int
    nn_atom: int
    zi: List[int]
    zb: List[float]
    energy_line: str


class CombinedResult(TypedDict):
    """TypedDict for CombinedResult structure"""
    n_conformers: int
    conformers: List[CombinedConformer]