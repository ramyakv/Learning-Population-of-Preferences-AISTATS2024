import sys
sys.path.append('../../')

import pickle
import numpy as np
from typing import List
from src.polytope.region import Region
from dataclasses import dataclass

@dataclass
class Item:
    lb: float
    ub: float
    m: int
    d: int
    item_seed: int
    n_h: int
    h_seed: int
    items: np.ndarray
    faces: np.ndarray
    pairs: np.ndarray
    full_regions: List[Region]
    regions: List[Region]
    hyperplanes: np.ndarray
    M: np.ndarray
    L: np.ndarray
    A: np.ndarray

    def save(self):
        filename = f"item_{self.lb}_{self.ub}_{self.m}_{self.d}_{self.item_seed}_{self.n_h}_{self.h_seed}.item"
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)