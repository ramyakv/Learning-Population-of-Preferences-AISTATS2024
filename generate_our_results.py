import pickle
import argparse
import numpy as np
import os
import tarfile
import shutil
from glob import glob
from time import time
from src.simulation.item import Item
from src.simulation.user import User
from src.simulation.result import Result
from src.polytope.region import estimate_p_hat, find_bounds, kl_bounds
from src.simulation.metric import total_variation, wasserstein


def our_method(lamb: float,
               epsilon: float,
               method: str,
               n_h: int,
               n_p: int,
               delta: float,
               q_true: np.ndarray,
               q_hat: np.ndarray,
               p_true: np.ndarray,
               M: np.ndarray,
               L: np.ndarray) -> dict:
    tik = time()
    p_hat_q_hat, status_q_hat = estimate_p_hat(M, q_hat, L, lamb=lamb, epsilon=epsilon, method=method)
    tok = time()
    time_elapsed = tok - tik

    p_hat_q_true, status_q_true = estimate_p_hat(M, q_true, L, lamb=lamb, epsilon=epsilon, method=method)

    p_lower, p_upper = find_bounds(q_hat, p_hat_q_hat, M, n_h, n_p, delta)
    p_lower_kl, p_upper_kl = kl_bounds(q_hat, delta, n_p, n_h)

    tv_err_q_hat = total_variation(p_true, p_hat_q_hat)
    tv_err_q_true = total_variation(p_true, p_hat_q_true)
    wa_err_q_hat = wasserstein(M, p_hat_q_hat, p_true)
    wa_err_q_true = wasserstein(M, p_hat_q_true, p_true)

    return {'time_elapsed': time_elapsed,
            'p_hat_q_hat': p_hat_q_hat,
            'status_q_hat': status_q_hat,
            'p_hat_q_true': p_hat_q_true,
            'status_q_true': status_q_true,
            'p_lower': p_lower,
            'p_upper': p_upper,
            'p_lower_kl': p_lower_kl,
            'p_upper_kl': p_upper_kl,
            'tv_err_q_hat': tv_err_q_hat,
            'tv_err_q_true': tv_err_q_true,
            'wa_err_q_hat': wa_err_q_hat,
            'wa_err_q_true': wa_err_q_true} 
    

def cla() -> argparse.Namespace:
    parser = argparse.ArgumentParser("[Pairwise Comparison] Generate simulation results")
    parser.add_argument("--lamb", default=1, type=float, help="lambda, the regularization parameter")
    parser.add_argument("--epsilon", default=1e-6, type=float, help="epsilon, the constraint parameter")
    parser.add_argument("--method", default="least-square-graph", type=str, help="optimization method")
    parser.add_argument("--delta", default=0.1, type=float, help="delta, the bound parameter")
    parser.add_argument("--idx", default=None, type=int, help="idx of the inputfile in the tarball")
    args = parser.parse_args()
    return args


def core(filename, args):
    with open(filename, "rb") as fp:
        user = pickle.load(fp)

    result = our_method(args.lamb, args.epsilon, args.method, user.n_h, user.n_p, args.delta, user.q_true, user.q_hat, user.p_true, user.M, user.L)
    result_obj = Result(lb=user.lb, ub=user.ub, m=user.m, d=user.d, item_seed=user.item_seed, n_h=user.n_h, h_seed=user.h_seed, user_seed=user.user_seed, user_dist=user.user_dist, n_p=user.n_p, noise=user.noise, lamb=args.lamb, epsilon=args.epsilon, method=args.method, delta=args.delta, M=user.M, L=user.L, p_true=user.p_true, q_true=user.q_true, q_hat=user.q_hat, V=user.V, **result)
        
    result_obj.save()

    
def run_all(args):
    files = glob("users/*.user")
    for i, filename in enumerate(files):
        print(f"====== {i} / {len(files) + 1} :: our results =====")
        core(filename, args)
    
    if not os.path.exists("results"):
        os.mkdir("results")
    for file in glob("*.result"):
        shutil.move(file, os.path.join("results", file))

    tarball = tarfile.open(f"results_our_{args.lamb}_{args.epsilon}_{args.method}_{args.delta}.tar.gz", "w:gz")
    tarball.add("./results")
    tarball.close()


def main():
    args = cla()
    if args.idx is None:
        run_all(args=args)
    else:
        filename = sorted(glob("users/*.user"))[args.idx]
        core(filename=filename, args=args)

    
if __name__ == '__main__':
    main()