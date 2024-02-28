import numpy as np
import cvxpy as cp


def total_variation(p_true: np.ndarray, p_hat: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(p_true.reshape(-1) - p_hat.reshape(-1), ord=1)


def wasserstein(M: np.ndarray, p_hat, p_true):
    C = np.linalg.norm(M[:, None, :] - M[:, :, None], ord=1, axis=0)
    C /= C.max()
    n = len(p_hat) 
    p_hat = p_hat.reshape((n, 1))
    p_true = p_true.reshape((n, 1))
    B = cp.Variable((n, n))
    objective = cp.Minimize(cp.trace(C.T @ B))
    constraints = [B @ np.ones((n, 1)) == p_true,
                   B.T @ np.ones((n, 1)) == p_hat,
                   B >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False, solver="GUROBI", reoptimize=True) 
    return result