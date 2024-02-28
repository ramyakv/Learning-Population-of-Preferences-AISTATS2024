import pickle
import numpy as np
from dataclasses import dataclass

@dataclass
class Result:
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
    lamb: float
    epsilon: float
    method: str
    delta: float
    M: np.ndarray
    L: np.ndarray
    p_true: np.ndarray
    q_true: np.ndarray
    q_hat: np.ndarray
    V: list
    time_elapsed: float
    p_hat_q_hat: np.ndarray
    p_hat_q_true: np.ndarray
    p_lower: float
    p_upper: float
    p_lower_kl: float
    p_upper_kl: float
    tv_err_q_hat: float
    tv_err_q_true: float
    wa_err_q_hat: float
    wa_err_q_true: float
    status_q_hat: str
    status_q_true: str

    def save(self):
        filename = f"result_{self.lb}_{self.ub}_{self.m}_{self.d}_{self.item_seed}_{self.n_h}_{self.h_seed}_{self.user_seed}_{self.user_dist}_{self.n_p}_{self.noise}_{self.lamb}_{self.epsilon}_{self.method}_{self.delta}.result"
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)


@dataclass
class LuResult:
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
    method: str
    M: np.ndarray
    L: np.ndarray
    p_true: np.ndarray
    q_true: np.ndarray
    q_hat: np.ndarray
    V: list
    A: np.ndarray
    tv_err_q_hat: float
    tv_err_q_true: float
    wa_err_q_hat: float
    wa_err_q_true: float
    em_seed: int
    p_hat: np.ndarray
    pi_s: np.ndarray
    sigma_s: list
    phi_s: np.ndarray
    K: int

    def save(self):
        filename = f"result_{self.lb}_{self.ub}_{self.m}_{self.d}_{self.item_seed}_{self.n_h}_{self.h_seed}_{self.user_seed}_{self.user_dist}_{self.n_p}_{self.noise}_{self.method}_{self.K}_{self.em_seed}.result"
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)