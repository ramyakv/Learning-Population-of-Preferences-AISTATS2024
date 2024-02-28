import sys
sys.path.append('lu')

import os
import pickle
import shutil
import tarfile
import argparse
import numpy as np
from lu import em
from glob import glob
from src.simulation.item import Item
from src.simulation.user import User
from src.simulation.result import LuResult
from src.simulation.metric import total_variation, wasserstein
from ranky.metric import kendall_tau_distance


def lu_method(V, K, A, p_true, items, regions, M, full_regions, em_seed):
    result = em(V, K, A, 10, 10, 1e-8, 10, em_seed)
    pi_s, sigma_s, phi_s = result['pi'], result['sigma'], result['phi']
    p_hat = np.zeros(len(regions))

    rankings = []
    for region in full_regions:
        center = region.center
        distances = np.linalg.norm(center - items, axis=1)
        ranking = np.argsort(distances)
        rankings.append(ranking)

    for i, sigma in enumerate(sigma_s):
        for j, ranking in enumerate(rankings):
            if np.all(ranking == sigma):
                for e, region in enumerate(regions):
                    if full_regions[j].center in region:
                        p_hat[e] += pi_s[i]
                        break
                break
        else:
            j = np.argmin([kendall_tau_distance(sigma, ranking) for ranking in rankings])
            for e, region in enumerate(regions):
                if full_regions[j].center in region:
                    p_hat[e] += pi_s[i]
                    break
    
    tv = total_variation(p_true, p_hat)
    wa = wasserstein(M, p_hat, p_true)

    return p_hat, tv, wa, pi_s, sigma_s, phi_s


def cla() -> argparse.Namespace:
    parser = argparse.ArgumentParser("[Pairwise Comparison] Generate simulation results")
    parser.add_argument("--K", default=5, type=int, help="number of clusters")
    parser.add_argument("--em_seed", default=0, type=np.int64, help="seed for EM algorithm")
    parser.add_argument("--filename", default=None, type=str, help="path to the user file")
    parser.add_argument("--idx", default=None, type=int, help="idx of the inputfile in the tarball")
    args = parser.parse_args()
    return args


def core(filename, args):
    with open(filename, "rb") as fp:
        user = pickle.load(fp)
        p_hat, tv, wa, pi_s, sigma_s, phi_s = lu_method(user.V, args.K, user.A, user.p_true, user.items, user.regions, user.M, user.full_regions, args.em_seed)
        LuResult(lb=user.lb, ub=user.ub, m=user.m, d=user.d, item_seed=user.item_seed, n_h=user.n_h, h_seed=user.h_seed, user_seed=user.user_seed, user_dist=user.user_dist, n_p=user.n_p, noise=user.noise, method="lu", M=user.M, L=user.L, p_true=user.p_true, q_true=user.q_true, q_hat=user.q_hat, V=user.V, A=user.A, tv_err_q_hat=tv, tv_err_q_true=tv, wa_err_q_hat=wa, wa_err_q_true=wa, em_seed=args.em_seed, p_hat=p_hat, pi_s=pi_s, sigma_s=sigma_s, phi_s=phi_s, K=args.K).save()

        
def run_all(args):
    files = glob("users/*.user")
    for i, filename in enumerate(files):
        print(f"====== {i} / {len(files) + 1} :: lu results =====")
        core(filename, args)
    
    if not os.path.exists("results"):
        os.mkdir("results")
    for file in glob("*.result"):
        shutil.move(file, os.path.join("results", file))

    tarball = tarfile.open(f"results_lu_{args.K}_{args.em_seed}.tar.gz", "w:gz")
    tarball.add("./results")
    tarball.close()


def main():
    args = cla()
    if args.idx is None:
        run_all(args)
    else:
        filename = sorted(glob("users/*.user"))[args.idx]
        core(filename, args)


    
if __name__ == '__main__':
    main()