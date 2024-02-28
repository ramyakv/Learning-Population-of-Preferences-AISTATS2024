import argparse
import numpy as np
from typing import Tuple, List
from itertools import combinations
from src.polytope.region import create_faces, create_M_matrix, Region
from src.polytope.region import generate_regions, calculate_hyperplanes, create_graph_laplacian_matrix
from src.simulation.item import Item


def create_items(lb: float, ub: float, m: int, d: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    item_rng = np.random.default_rng(seed)
    items = item_rng.uniform(lb, ub, size=(m, d)).astype(
        np.float32
    )  
    return items


def create_hyperlanes(items: np.ndarray, pairs: np.ndarray, faces: np.ndarray, n_h: int, seed: int) -> Tuple[np.ndarray, List[Region], np.ndarray, List[Region]]:
    full_hyperplanes = calculate_hyperplanes(items, pairs)
    full_regions = generate_regions(full_hyperplanes, faces)

    # sample a subset of hyperplanes
    if n_h != int(len(items) * (len(items) - 1) / 2):
        h_rng = np.random.default_rng(seed)
        pairs = pairs[h_rng.choice(np.arange(len(pairs)), size=n_h, replace=False)]

    hyperplanes = calculate_hyperplanes(items, pairs)
    regions = generate_regions(hyperplanes, faces)
    return full_hyperplanes, full_regions, hyperplanes, regions 


def cla() -> argparse.Namespace:
    parser = argparse.ArgumentParser("[Pairwise Comparison] Generate items for simulation")
    parser.add_argument("--lb", default=-1, type=float, help="lower bound of items")
    parser.add_argument("--ub", default=1, type=float, help="upper bound of items")
    parser.add_argument("--m", default=5, type=int, help="number of items")
    parser.add_argument("--d", default=2, type=int, help="dimension of items")
    parser.add_argument("--item_seed", default=0, type=int, help="seed for randomly generating items from a uniform distribution")
    parser.add_argument("--n_h", default=-1, type=int, help="number of hyperplanes -1 means all hyperplanes")
    parser.add_argument("--h_seed", default=0, type=int, help="seed for randomly choosing hyperplanes")
    parser.add_argument("--idx", default=1, type=int, help="idx of the inputfile in the tarball")
    args = parser.parse_args()

    if args.n_h == -1:
        args.n_h = int(args.m * (args.m - 1) / 2)

    return args


def main():
    args = cla()
    items = create_items(args.lb, args.ub, args.m, args.d, args.item_seed)
    faces = create_faces(args.lb, args.ub, args.d)
    pairs = np.array(list(map(list, combinations(range(args.m), 2))))
    _ , full_regions, hyperplanes, regions = create_hyperlanes(items, pairs, faces, args.n_h, args.h_seed)
    M, _ = create_M_matrix(regions, hyperplanes)
    L = create_graph_laplacian_matrix(regions, M, lambda x: 1 / x)
    A = np.arange(args.m)

    item_obj = Item(
        lb=args.lb,
        ub=args.ub,
        m=args.m,
        d=args.d,
        item_seed=args.item_seed,
        n_h=args.n_h,
        h_seed=args.h_seed,
        items=items,
        faces=faces,
        pairs=pairs,
        full_regions=full_regions,
        regions=regions,
        hyperplanes=hyperplanes,
        M=M,
        L=L,
        A=A)
    item_obj.save() 


if __name__ == '__main__':
    main()