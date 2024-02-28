import sys
sys.path.append('../../')

import pickle
import numpy as np
from typing import List
from src.polytope.region import Region
from dataclasses import dataclass

@dataclass    
class User:
    lb: float
    ub: float
    m: int
    d: int
    item_seed: int
    n_h: int
    h_seed: int
    user_seed: int
    user_dist: str
    n_p: int
    noise: str
    items: np.ndarray
    M: np.ndarray
    L: np.ndarray
    full_regions: List[Region]
    regions: List[Region]
    p_true: np.ndarray
    q_true: np.ndarray
    q_hat: np.ndarray
    V: list
    A: np.ndarray

    def save(self):
        filename = f"user_{self.lb}_{self.ub}_{self.m}_{self.d}_{self.item_seed}_{self.n_h}_{self.h_seed}_{self.user_seed}_{self.user_dist}_{self.n_p}_{self.noise}.user"
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)