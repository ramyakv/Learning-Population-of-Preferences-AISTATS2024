import os
import shutil
import pickle
import tarfile
import argparse
from glob import glob
from src.polytope.region import calculate_p_true, calculate_q_true, estimate_q_hat
from src.simulation.item import Item
from src.simulation.user import User
from src.distribution.distribution import (
    Uniform,
    Gaussian,
    Mix2Gaussian,
    Mix3Gaussian,
    TwoPoints,
    ThreePoints,
    FourPoints,
    UserDistribution
)


def get_user_distribution(lb, ub, d, user_seed, user_dist) -> UserDistribution:
    if user_dist == "uniform":
        return Uniform(lb, ub, d, user_seed)
    elif user_dist == "gaussian":
        return Gaussian(lb, ub, d, user_seed)
    elif user_dist == "mix2gaussian":
        return Mix2Gaussian(lb, ub, d, user_seed)
    elif user_dist == "mix3gaussian":
        return Mix3Gaussian(lb, ub, d, user_seed)
    elif user_dist == "twopoints":
        return TwoPoints(lb, ub, d, user_seed)
    elif user_dist == "threepoints":
        return ThreePoints(lb, ub, d, user_seed)
    elif user_dist == "fourpoints":
        return FourPoints(lb, ub, d, user_seed)
    else:
        raise ValueError("No such user distribution.")


def cla() -> argparse.Namespace:
    parser = argparse.ArgumentParser("[Pairwise Comparison] Generate users for simulation")
    parser.add_argument("--user_seed", default=0, type=int, help="random seed for sampling users")
    parser.add_argument("--user_dist", default="uniform", type=str, help="user distribution")
    parser.add_argument("--n_p", default=1000, type=int, help="number of users per pair")
    parser.add_argument("--noise", default="noiseless", type=str, help="noise when estimating q")
    parser.add_argument("--idx", default=1, type=int, help="idx of the inputfile in the tarball")
    args = parser.parse_args()
    return args


def main():
    args = cla()


    files = glob("items/*.item")
    for i, filename in enumerate(files):
        with open(filename, "rb") as fp:
            item = pickle.load(fp)

            user_dist = get_user_distribution(item.lb, item.ub, item.d, args.user_seed, args.user_dist)
            users = user_dist.samples(item.n_h * args.n_p, replace=False)
            p_true, _ = calculate_p_true(item.regions, user_dist.points)
            q_true = calculate_q_true(item.M, p_true)
            q_hat, V = estimate_q_hat(
                item.hyperplanes,
                users,
                n_p=args.n_p,
                noise=args.noise,
                pairs=item.pairs,
                items=item.items,
            )
            user_obj = User(
                lb=item.lb,
                ub=item.ub,
                m=item.m,
                d=item.d,
                item_seed=item.item_seed,
                n_h=item.n_h,
                h_seed=item.h_seed,
                user_seed=args.user_seed,
                user_dist=args.user_dist,
                n_p=args.n_p,
                noise=args.noise,
                items=item.items,
                M=item.M,
                L=item.L,
                regions=item.regions,
                full_regions=item.full_regions,
                p_true=p_true,
                q_true=q_true,
                q_hat=q_hat,
                V=V,
                A=item.A
            )
            user_obj.save()

    if not os.path.exists("users"):
        os.mkdir("users")
    for file in glob("*.user"):
        shutil.move(file, os.path.join("users", file))

    tarball = tarfile.open(f"users_{args.user_seed}_{args.user_dist}_{args.n_p}_{args.noise}.tar.gz", "w:gz")
    tarball.add("./users")
    tarball.close()

    
if __name__ == '__main__':
    main()