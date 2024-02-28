from __future__ import annotations
import numpy as np
import cvxpy as cp
from time import time
from itertools import combinations
from typing import Optional, List, Tuple, Callable
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, _qhull
from collections import defaultdict


class Region:
    hyperplanes: Optional[np.ndarray]
    center: Optional[np.ndarray]
    left_of: set
    id: int
    _next_id = 0

    def __init__(
        self, hyperplanes: Optional[np.ndarray], center: Optional[np.ndarray]
    ) -> Region:
        self.hyperplanes = hyperplanes
        self.center = center
        self.id = Region._next_id
        self.left_of = set()
        Region._next_id += 1

    def __hash__(self) -> int:
        return self.id

    def __contains__(self, point: np.ndarray):
        """Check if the point is in this region
        """
        return np.all(
            np.dot(self.hyperplanes[:, :-1], point) + self.hyperplanes[:, -1] < 0
        )

    def __repr__(self) -> str:
        return str(self.id) + " " + str(self.center)

    def mitosis(self, h: np.ndarray, h_o: np.ndarray, p: np.ndarray, p_o: np.ndarray):
        r = Region(hyperplanes=np.vstack((self.hyperplanes, h)), center=p)
        r_o = Region(hyperplanes=np.vstack((self.hyperplanes, h_o)), center=p_o)
        return r, r_o

    def volume(self) -> float:
        try:
            hi = HalfspaceIntersection(self.hyperplanes, self.center)
        except _qhull.QhullError:
            # degenerate
            return 0
        try:
            vol = ConvexHull(hi.intersections).volume
            return vol
        except _qhull.QhullError:
            # degenerate
            return 0


def create_faces(lb: int, ub: int, d: int) -> np.ndarray:
    """Create the polytopes (halfspaces) that defines the bounding box
    """
    cell = np.vstack([np.repeat(lb, d), np.repeat(ub, d)])
    vtxs = []
    for i in range(cell.shape[1]):
        l_vtx, u_vtx = np.copy(cell[0]), np.copy(cell[1])

        l_vtx[i], u_vtx[i] = u_vtx[i], l_vtx[i]

        vtxs.append(l_vtx)
        vtxs.append(u_vtx)

    vtxs.append(cell[0])
    vtxs.append(cell[1])
    vtxs = np.array(vtxs)

    faces = np.unique(ConvexHull(vtxs).equations, axis=0)
    return faces


def K(j, M):
    # K_j be the position of rows of M whose jth entry is 1.
    k = set()
    for i, row in enumerate(M):
        if row[j] == 1:
            k.add(i)
    return list(k)


def Q_j(p, j, q, M):
    q_j = np.zeros(p.shape)
    for i in range(len(q_j)):
        if i != j:
            q_j[i] = q[K(i, M)].min()
        else:
            q_j[i] = p[i]
    return q_j


def Q_j_0(p, j, q, M):
    q_j = np.zeros(p.shape)
    for i in range(len(q_j)):
        if i != j:
            q_j[i] = q[K(i, M)].min()
        else:
            q_j[i] = 0
    return q_j


def kl_bounds(q_hat, delta, n_p, n_h):
    q_hat = q_hat.reshape(-1)
    delta_prime = delta / (4 * n_h)

    upper_bound = []
    for i in range(len(q_hat)):
        q_hat_aibi = q_hat[i]
        qup = cp.Variable(1)
        objective = cp.Maximize(qup)
        constraints = [qup <= 1,
                       qup >= q_hat_aibi,
                       -q_hat_aibi * cp.log(qup / q_hat_aibi) - (1 - q_hat_aibi) * cp.log((1 - qup) / (1 - q_hat_aibi)) <= cp.log(1 / delta_prime) / n_p]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        upper_bound.append(qup.value)
    
    lower_bound = []
    for i in range(len(q_hat)):
        q_hat_aibi = q_hat[i]
        qlow = cp.Variable(1)
        objective = cp.Maximize(qlow)
        constraints = [qlow >= 0,
                       qlow <= q_hat_aibi, 
                       -q_hat_aibi * cp.log(qlow / q_hat_aibi) - (1 - q_hat_aibi) * cp.log((1 - qlow) / (1 - q_hat_aibi)) <= cp.log(1 / delta_prime) / n_p]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        lower_bound.append(qlow.value)
    
    return np.array(lower_bound), np.array(upper_bound)
    


def find_bounds(q_hat, p_hat, M, n_h, n, delta):
    Mf = np.vstack([M[:-1], 1 - M[:-1]])
    c = np.sqrt(np.log(4 * n_h / delta) / (2 * n))
    q_hat = q_hat.reshape(-1)
    p_hat = p_hat.reshape(-1)
    q_hat = np.hstack([q_hat[:-1], 1 - q_hat[:-1]])
    p_upper = np.zeros(p_hat.shape)
    p_lower = np.zeros(p_hat.shape)
    for j in range(len(p_hat)):
        lower_bound = 0
        upper_bound = 1
        for i in K(j, Mf):
            cand = (
                q_hat[i]
                - np.dot(Mf[i], Q_j_0(p_hat, j, q_hat, Mf))
                - (np.linalg.norm(Mf[i], 1) + 1) * c
            )
            lower_bound = max(lower_bound, cand)
            upper_bound = min(upper_bound, q_hat[i] + c)
        p_upper[j] = upper_bound
        p_lower[j] = lower_bound
    return p_lower, p_upper


def calculate_hyperplanes(items: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Compute the hyperplanes that orthogonal bisect each pair of items

    Args:
        items (np.ndarray): items
        pairs (np.ndarray): pairs, where l, r = pair[i];
        l and r are indices of items of the pair in items

    Returns:
        np.ndarray: each row is a hyperplane. The first d dimension is the
        normal vector a and the last dimension is the offset b.
        A row represents the hyperplane a^Tx + b = 0.
    """
    d = items.shape[1]
    h_s = np.zeros((len(pairs), d + 1))
    for i in range(len(pairs)):
        l_pt_idx, r_pt_idx = pairs[i]
        l_pt, r_pt = items[l_pt_idx], items[r_pt_idx]
        m_pt = (r_pt + l_pt) / 2
        a = (r_pt - l_pt) / np.linalg.norm(r_pt - l_pt)
        b = -np.dot(m_pt, a)
        res = np.hstack((a.reshape(-1), np.array(b).reshape(-1)))
        h_s[i] = res
    return h_s


def chebyshev_center(halfspaces):
    norm_vector = np.reshape(
        np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1)
    )
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    if res.success:
        return res.x[:-1]
    else:
        return None


def check_intersection(h: np.ndarray, h_o: np.ndarray, region: Region):
    p = chebyshev_center(np.vstack((region.hyperplanes, h)))
    p_o = chebyshev_center(np.vstack((region.hyperplanes, h_o)))

    if p is None or p_o is None:
        return False, None, None
    if p in region and p_o in region:
        return True, p, p_o
    else:
        return False, None, None


def generate_regions(hyperplanes: np.ndarray, faces: np.ndarray) -> List[Region]:
    regions = [Region(hyperplanes=faces, center=None)]

    for h in hyperplanes:
        # hyperplane with normal vector that points left
        h_o = -1 * h
        # to track which node is cut by h
        intersected_regions = []

        for region in regions:
            is_intersected, p, p_o = check_intersection(h, h_o, region)

            if is_intersected:
                intersected_regions.append((region, p, p_o))

        for region, p, p_o in intersected_regions:
            r, r_o = region.mitosis(h, h_o, p, p_o)
            if r.volume() <= 1e-12 or r_o.volume() <= 1e-12:
                continue
            regions.remove(region)
            regions.append(r)
            regions.append(r_o)

    return regions


def create_M_matrix(regions: List[Region], hyperplanes: List[np.ndarray]) -> np.ndarray:
    for region in regions:
        for h_id, h in enumerate(hyperplanes):
            a, b = h[:-1], h[-1]
            if np.dot(region.center, a) + b <= 0:
                region.left_of.add(h_id)

    num_regions = len(regions)
    num_hyperplanes = len(hyperplanes)
    M = np.zeros((num_hyperplanes + 1, num_regions))

    h_to_nodes = defaultdict(lambda: [])
    for region_idx, region in enumerate(regions):
        for hyperplane_idx in region.left_of:
            h_to_nodes[hyperplane_idx].append(region_idx)

    for hyperplane_idx, region_indices in h_to_nodes.items():
        M[hyperplane_idx][np.array(region_indices)] = 1
    M[-1] = 1
    Mf = np.zeros((num_hyperplanes * 2, num_regions))
    Mf[:num_hyperplanes] = M[:-1]
    Mf[num_hyperplanes:] = 1 - M[:-1]
    return M, Mf


def create_graph_laplacian_matrix(
    regions: List[Region], M: np.ndarray, inverse_function: Callable
):
    one_over_area = 1 / np.array(list(map(lambda x: x.volume(), regions)))
    one_over_area = one_over_area / one_over_area.sum()

    W = np.zeros((M.shape[1], M.shape[1]))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = (M[:, i] != M[:, j]).sum()
            if W[i, j] != 0:
                W[i, j] = inverse_function(W[i, j])
    D = np.diag(np.sum(W, 0))
    L = D - W

    A = np.diag(one_over_area)
    L = A.T @ L @ A

    return L


def calculate_p_true(
    regions: List[Region], users: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    num_points_in = np.zeros(len(regions))

    for region_idx, region in enumerate(regions):
        Ab = region.hyperplanes
        A = Ab[:, :-1]
        b = Ab[:, -1].reshape(-1, 1)

        where = np.all(((A @ users.T) + b <= 0), axis=0)
        num_points_in[region_idx] = where.sum()

    p_true = num_points_in / num_points_in.sum()
    return p_true.reshape(-1, 1), num_points_in


def calculate_q_true(M: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    return M @ p_true


def estimate_q_hat_from_experiment(pairs: np.ndarray, user_set: str):
    import pandas as pd

    df = pd.read_csv(user_set)

    q_hat = []
    for left, right in pairs:
        matched = df[(df["left"] == left) & (df["right"] == right)]
        q_hat.append((matched["answer"] == 0).sum())

    q_hat.append(100)

    return (np.array(q_hat) / 100).reshape(-1, 1)


def estimate_q_hat(
    hyperplanes: np.ndarray,
    sampled_users: np.ndarray,
    n_p: int,
    noise: str,
    pairs: np.ndarray,
    items: np.ndarray,
):
    V = []
    q_hat = np.zeros(len(hyperplanes) + 1)
    for i, hyperplane in enumerate(hyperplanes):
        u = sampled_users[i * n_p : (i + 1) * n_p]
        a, b = hyperplane[:-1], hyperplane[-1]

        if noise == "noiseless":
            votes = np.dot(u, a) + b <= 0
            q_hat[i] = votes.mean()
        elif noise == "bernoulli01":
            votes = np.dot(u, a) + b <= 0
            probs = np.random.uniform(size=len(votes))
            votes[probs < 0.01] = 1 - votes[probs < 0.01]
            q_hat[i] = votes.mean()
        elif noise == "bernoulli02":
            votes = np.dot(u, a) + b <= 0
            probs = np.random.uniform(size=len(votes))
            votes[probs < 0.02] = 1 - votes[probs < 0.02]
            q_hat[i] = votes.mean()
        elif noise == "bernoulli05":
            votes = np.dot(u, a) + b <= 0
            probs = np.random.uniform(size=len(votes))
            votes[probs < 0.05] = 1 - votes[probs < 0.05]
            q_hat[i] = votes.mean()
        elif "sigmoid" in noise:
            votes = np.dot(u, a) + b <= 0
            q_hat[i] = votes.mean()

            l_idx, r_idx = pairs[i]
            l_pt, r_pt = items[l_idx], items[r_idx]

            d_left = np.linalg.norm(u - l_pt, axis=1)
            d_right = np.linalg.norm(u - r_pt, axis=1)
            d_diff = -np.abs(d_left - d_right)

            if noise == "sigmoid300":
                flipping_probs = 1 / (1 + np.exp(-300 * d_diff))
            elif noise == "sigmoid500":
                flipping_probs = 1 / (1 + np.exp(-500 * d_diff))
            else:
                raise ValueError(f"{noise = } does not exist.")

            probs = np.random.uniform(size=len(flipping_probs))
            votes[probs < flipping_probs] = 1 - votes[probs < flipping_probs]
            q_hat[i] = votes.mean()
        else:
            raise ValueError(f"{noise = } does not exist.")

        left, right = pairs[i]
        for vote in votes:
            if vote == 1:
                V.append([f"{left} > {right}"])
            else:
                V.append([f"{right} > {left}"])

    q_hat[-1] = 1
    return q_hat.reshape(-1, 1), V


def egd(M, q, rate0, tol, maxIters, lamb, L):
    q = q.reshape(-1, 1)
    p_hat = np.ones(len(L)) / len(L)
    p_hat = p_hat.reshape(-1, 1)

    for i in range(1, maxIters + 1):
        coef = np.sqrt(1 / (30 + i))
        rate = rate0 * coef

        g = -(M.T @ (M @ p_hat - q) + lamb * L @ p_hat)

        p_new = p_hat * np.exp(rate * g)
        p_new = p_new / p_new.sum()

        if np.linalg.norm(p_new - p_hat) < tol:
            break

        p_hat = p_new
    return p_hat.reshape(-1)


def estimate_p_hat(M: np.ndarray, q: np.ndarray, L: np.ndarray, lamb: float, epsilon=1e-12, method="least-square-graph"):
    p_hat = cp.Variable((M.shape[1], 1))
    if method == "least-square-graph":
        loss = cp.sum_squares(M @ p_hat - q) + lamb * cp.quad_form(
            p_hat, cp.atoms.affine.wraps.psd_wrap(L)
        )
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1]
    elif method == "least-square-l1":
        loss = cp.sum_squares(M @ p_hat - q) + lamb * cp.sum(p_hat)
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1]
    elif method == "least-square-l2":
        loss = cp.sum_squares(M @ p_hat - q) + lamb * cp.sum_squares(p_hat)
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1]
    elif method == "least-square":
        loss = cp.sum_squares(M @ p_hat - q)
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1]
    elif method == "kl":
        loss = cp.sum(cp.rel_entr(q, M @ p_hat))
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1]
    elif method == "graph-least-square" :
        loss = lamb * cp.quad_form(
            p_hat, cp.atoms.affine.wraps.psd_wrap(L)
        )
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1, cp.sum_squares(M @ p_hat - q) <= epsilon]
    elif method == "l1-least-square" :
        loss = lamb * cp.sum(p_hat)
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1, cp.sum_squares(M @ p_hat - q) <= epsilon]
    elif method == "l2-least-square" :
        loss = lamb * cp.sum_squares(p_hat)
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1, cp.sum_squares(M @ p_hat - q) <= epsilon]
    elif method == "graph-kl" :
        loss = lamb * cp.quad_form(
            p_hat, cp.atoms.affine.wraps.psd_wrap(L)
        )
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1, cp.sum(cp.rel_entr(q, M @ p_hat)) <= epsilon]
    elif method == "l1-kl" :
        loss = lamb * cp.sum(p_hat)
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1, cp.sum(cp.rel_entr(q, M @ p_hat)) <= epsilon]
    elif method == "l2-kl" :
        loss = lamb * cp.sum_squares(p_hat)
        constraints = [p_hat >= 0, cp.sum(p_hat) == 1, cp.sum(cp.rel_entr(q, M @ p_hat)) <= epsilon]
    else:
        raise ValueError("No such method")
    
    prob = cp.Problem(
        cp.Minimize(loss),
        constraints,
    )

    prob.solve(verbose=False,
               solver=cp.ECOS)
    # if prob.status != cp.OPTIMAL:
    #     print(p_hat.value)
    #     raise Exception("Solver did not converge!")
    return p_hat.value, prob.status