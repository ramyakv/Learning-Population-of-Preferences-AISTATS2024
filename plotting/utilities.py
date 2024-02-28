import sys
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
sys.path.append('..')
from src.polytope.region import generate_regions, create_faces, calculate_hyperplanes, create_M_matrix
from src.polytope.region import create_graph_laplacian_matrix, estimate_p_hat, find_bounds
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial import convex_hull_plot_2d, ConvexHull, HalfspaceIntersection

def total_variation(p_true: np.ndarray, p_hat: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(p_true.reshape(-1) - p_hat.reshape(-1), ord=1)


def wasserstein(M: np.ndarray, p_hat, p_true):
    C = np.linalg.norm(M[:, None, :] - M[:, :, None], ord=1, axis=0)
    C /= C.max()
    n = len(p_hat) 
    B = cp.Variable((n, n))
    objective = cp.Minimize(cp.trace(C.T @ B))
    constraints = [B @ np.ones((n, 1)) == p_true,
                   B.T @ np.ones((n, 1)) == p_hat,
                   B >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False, solver="GUROBI", reoptimize=True) 
    return result


def setup(n_h, seed, items_path="./zappos-5-embedding-2d.npy"):
    np.random.seed(seed)
    items = np.load(items_path)
    fixed_items = np.zeros(items.shape) 
    fixed_items[0] = items[-1]
    fixed_items[1:]= items[:-1]
    items = fixed_items

    faces = create_faces(-1, 1, 2)
    pairs = np.array(list(map(list, combinations(range(5), 2))))
    if n_h != len(pairs):
        idx = np.random.choice(np.arange(len(pairs)), size=n_h, replace=False)
        pairs = pairs[idx]
    hyperplanes = calculate_hyperplanes(items, pairs)
    regions = generate_regions(hyperplanes, faces)
    M, _ = create_M_matrix(regions, hyperplanes)
    L = create_graph_laplacian_matrix(regions,
                                      M,
                                      lambda x: 1 / x)
    return hyperplanes, regions, pairs, M, L


def p_star_from_workers(prop, df, hyperplanes, regions, pairs, seed):
    p_star_df = defaultdict(list)
    n_workers = len(df.index.unique())
    # workers used to generate p_star
    np.random.seed(seed)
    workers_idx = np.random.choice(df.index.unique(), size=int(n_workers * prop), replace=False)
    # the remaining workers, to be used to learn p_hat
    agg_bin = np.zeros(len(regions))
    pair_to_h_id = {tuple(pair): i for i, pair in enumerate(pairs)}
    for worker_idx in workers_idx:
        worker_bin = np.zeros(len(regions))
        for _, q in df.loc[worker_idx].iterrows():
            if not np.any(np.all(pairs == [q.left, q.right], axis=1)):
                continue
            h_id = pair_to_h_id[q.left, q.right]
            h = hyperplanes[h_id]
            if q.answer == 1:
                h = h * -1
            for i, region in enumerate(regions):
                a, b = h[:-1], h[-1]
                if np.dot(region.center, a) + b <= 0:
                    worker_bin[i] += 1

        where = worker_bin == worker_bin.max()
        count = where.sum()
        agg_bin[where] += 1 / count

    agg_bin = agg_bin / agg_bin.sum()
    for i, val in enumerate(agg_bin):
        p_star_df["region ID"].append(i)
        p_star_df["prob"].append(val)
    return pd.DataFrame(p_star_df)


def p_star_from_workers_and_p_hat(prop, df, hyperplanes, regions, pairs, M, L, seed, method, lamb, repeat=1):
    p_star_df = defaultdict(list)
    p_hat_df = defaultdict(list)
    p_lower_df = defaultdict(list)
    p_upper_df = defaultdict(list)

    n_workers = len(df.index.unique())
    # workers used to generate p_star
    np.random.seed(seed)
    workers_idx = np.random.choice(df.index.unique(), size=int(n_workers * prop), replace=False)
    # the remaining workers, to be used to learn p_hat
    agg_bin = np.zeros(len(regions))
    pair_to_h_id = {tuple(pair): i for i, pair in enumerate(pairs)}
    for worker_idx in workers_idx:
        worker_bin = np.zeros(len(regions))
        for _, q in df.loc[worker_idx].iterrows():
            if not np.any(np.all(pairs == [q.left, q.right], axis=1)):
                continue
            h_id = pair_to_h_id[q.left, q.right]
            h = hyperplanes[h_id]
            if q.answer == 1:
                h = h * -1
            for i, region in enumerate(regions):
                a, b = h[:-1], h[-1]
                if np.dot(region.center, a) + b <= 0:
                    worker_bin[i] += 1

        where = worker_bin == worker_bin.max()
        count = where.sum()
        agg_bin[where] += 2 / count

    if prop > 0:
        agg_bin = agg_bin / agg_bin.sum()

    rest_idx = np.array(list(set(df.index.unique()) - set(workers_idx)))
    V = []
    for _ in range(repeat):
        np.random.shuffle(rest_idx)
        n_p = len(rest_idx) // len(pairs)
        q_hat = []
        for i, (left, right) in enumerate(pairs):
            p_idx = rest_idx[i * n_p:(i + 1) * n_p]
            p_df = df.loc[p_idx]
            p_df = p_df[(p_df.left == left) & (p_df.right == right)]
            for row in p_df.itertuples():
                if row.answer == 0:
                    V.append([f"{row.left} > {row.right}"])
                else:
                    V.append([f"{row.right} > {row.left}"])
            q_hat.append((p_df.answer == 0).sum())
        q_hat.append(n_p)
        q_hat = np.array(q_hat) / n_p
        q_hat = q_hat.reshape(-1, 1)
        p_hat = estimate_p_hat(M, q_hat, L, lamb, 1e-12, method)[0].reshape(-1)

        lower, upper = find_bounds(q_hat, p_hat, M, 10, n_p, 0.1)
        for i, val in enumerate(lower):
            p_lower_df["region ID"].append(i)
            p_lower_df["prob"].append(val)
        for i, val in enumerate(upper):
            p_upper_df["region ID"].append(i)
            p_upper_df["prob"].append(val)

        for i, val in enumerate(p_hat):
            p_hat_df["region ID"].append(i)
            p_hat_df["prob"].append(val)
            
    for i, val in enumerate(agg_bin):
        p_star_df["region ID"].append(i)
        p_star_df["prob"].append(val)
    return pd.DataFrame(p_star_df), pd.DataFrame(p_hat_df), method, seed, V, pd.DataFrame(p_lower_df), pd.DataFrame(p_upper_df)


def p_star_from_workers_and_bounds(prop, df, hyperplanes, regions, pairs, M, L, delta, n_h, seed, repeat=100):
    p_star_df = defaultdict(list)
    p_lower_df = defaultdict(list)
    p_upper_df = defaultdict(list)

    n_workers = len(df.index.unique())
    # workers used to generate p_star
    np.random.seed(seed)
    workers_idx = np.random.choice(df.index.unique(), size=int(n_workers * prop), replace=False)
    # the remaining workers, to be used to learn p_hat
    agg_bin = np.zeros(len(regions))
    pair_to_h_id = {tuple(pair): i for i, pair in enumerate(pairs)}
    for worker_idx in workers_idx:
        worker_bin = np.zeros(len(regions))
        for _, q in df.loc[worker_idx].iterrows():
            if not np.any(np.all(pairs == [q.left, q.right], axis=1)):
                continue
            h_id = pair_to_h_id[q.left, q.right]
            h = hyperplanes[h_id]
            if q.answer == 1:
                h = h * -1
            for i, region in enumerate(regions):
                a, b = h[:-1], h[-1]
                if np.dot(region.center, a) + b <= 0:
                    worker_bin[i] += 1

        where = worker_bin == worker_bin.max()
        count = where.sum()
        agg_bin[where] += 1 / count

    agg_bin = agg_bin / agg_bin.sum()

    rest_idx = np.array(list(set(df.index.unique()) - set(workers_idx)))
    for _ in range(repeat):
        np.random.shuffle(rest_idx)
        n_p = len(rest_idx) // len(pairs)
        q_hat = []
        for i, (left, right) in enumerate(pairs):
            p_idx = rest_idx[i * n_p:(i + 1) * n_p]
            p_df = df.loc[p_idx]
            p_df = p_df[(p_df.left == left) & (p_df.right == right)]
            q_hat.append((p_df.answer == 0).sum())
        q_hat.append(n_p)
        q_hat = np.array(q_hat) / n_p
        q_hat = q_hat.reshape(-1, 1)
        p_hat = estimate_p_hat(M, q_hat, L, 1)[0].reshape(-1)

        lower, upper = find_bounds(q_hat, p_hat, M, n_h, n_p, delta)
        for i, val in enumerate(lower):
            p_lower_df["region ID"].append(i)
            p_lower_df["prob"].append(val)
        for i, val in enumerate(upper):
            p_upper_df["region ID"].append(i)
            p_upper_df["prob"].append(val)

            
    for i, val in enumerate(agg_bin):
        p_star_df["region ID"].append(i)
        p_star_df["prob"].append(val)
    return pd.DataFrame(p_star_df), pd.DataFrame(p_lower_df), pd.DataFrame(p_upper_df)


def plot_embedding(regions, ax, p_hat):
    def get_image(path):
        return OffsetImage(plt.imread(path, format='jpg'), zoom=0.3, alpha=0.7)

    items = np.load("./zappos-5-embedding-2d.npy")
    fixed_items = np.zeros(items.shape) 
    fixed_items[0] = items[-1]
    fixed_items[1:]= items[:-1]
    items = fixed_items
    paths = [f"../dataset/{i}.jpg" for i in range(0, 5)]
    embedding = np.load("./zappos-50k-embedding-2d.npy")[:-5]
    lb = -1
    ub = 1

    for i, region in enumerate(regions):
        center = region.center
        hs = HalfspaceIntersection(region.hyperplanes, center)
        ch = ConvexHull(hs.intersections)
        convex_hull_plot_2d(ch, ax)
        center = hs.intersections.mean(axis=0)

        ax.text(
            center[0],
            center[1],
            f"{i}")#, {p_hat[p_hat["region ID"] == i].prob.median():.3f} {p_hat[p_hat["region ID"] == i].prob.min():.3f} {p_hat[p_hat["region ID"] == i].prob.max():.3f})


    for x, y, path in zip(items[:, 0], items[:, 1], paths):
        ab = AnnotationBbox(get_image(path), (x, y), frameon=False)
        ax.add_artist(ab)


    labels = pd.read_csv('annotations_file_all.csv', header=None)[1].values
    colors = list(mcolors.CSS4_COLORS.values())
    for c in np.unique(labels):
        ax.scatter(
            embedding[labels == c, 0],
            embedding[labels == c, 1],
            c=colors[c])
            #label=c)

    ax.scatter(items[:, 0], items[:, 1], c='black', marker='x')
    ax.set_xlim([lb, ub])
    ax.set_ylim([lb, ub])
    ax.set_xticks([lb, ub])
    ax.set_yticks([lb, ub])