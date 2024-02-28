from typing import List
from glob import glob
from scipy.io import loadmat
from src.simulation.metric import total_variation
from src.distribution.distribution import Uniform, Mix2Gaussian, Mix3Gaussian, Gaussian
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set(context='paper', style='white', palette='deep', font='sans-serif', font_scale=2, color_codes=False, rc=None)

def load_chtc_result(root_path: str, var_key: str):
    var_s = []
    noise_s = []
    user_dist_s = []
    tv_val_q_true_s = []
    tv_val_q_hat_s = []
    methods = []
    
    for file in glob(os.path.join(root_path, "*.mat")):
        mat = loadmat(file)
        user_dist = np.ravel(mat['user_dist'])[0]
        method = np.ravel(mat['method'])[0]
        noise = np.ravel(mat['noise'])[0]
        p_true = mat['p_true']
        p_hat_q_hat = mat['p_hat_q_hat']
        p_hat_q_true = mat['p_hat_q_true']

        tv_q_true_val = total_variation(p_true, p_hat_q_true)
        tv_q_hat_val = total_variation(p_true, p_hat_q_hat)

        var = np.ravel(mat[var_key])[0]

        user_dist_s.append(user_dist)
        noise_s.append(noise)
        tv_val_q_true_s.append(tv_q_true_val)
        tv_val_q_hat_s.append(tv_q_hat_val)
        var_s.append(var)
        methods.append(method)

    df = pd.DataFrame({"user_dist": user_dist_s,
                       "total_variation_q_hat": tv_val_q_hat_s,
                       "total_variation_q_true": tv_val_q_true_s,
                       "noise": noise_s,
                       "method": method,
                       var_key: var_s })
    return df


def plot_chtc_result(df: pd.DataFrame, var_key: str):
    g = sns.relplot(
        data=df,
        col='user_dist',
        kind='line',
        x=var_key,
        y='total_variation_q_hat',
        hue='noise',
        style='noise',
        err_style='bars',
        markers=True,
    )

    for ax in g.axes:
        for a in ax:
            a.set_ylim([-0.05, 1.05])
            a.set_ylabel(
                r"$\textrm{TV}(\hat{\textbf{p}}_{\mathcal{H}(S_m)}, \textbf{p}^*_{\mathcal{H}(S_m)})$",
                fontsize=20)
            a.set_xlabel(f"$({var_key})$", fontsize=24)
            a.tick_params(labelsize=20)

    plt.savefig(f"./fig/{var_key}_vs_tv.png", dpi=300)
    plt.show()
    

def plot_chtc_hyperplanes(root_path: str, n_h: int):
    h_s = {}
    for file in glob(os.path.join(root_path, "*.mat")):
        mat = loadmat(file)
        h = mat["hyperplanes"]
        if mat["n_h"][0, 0] == n_h:
            h_s[mat["h_seed"][0, 0]] = h

    dists = list(map(lambda x: x.samples(100, True), [Mix3Gaussian(-1, 1, 2, 0),
                                                        Mix2Gaussian(-1, 1, 2, 0),
                                                        Uniform(-1, 1, 2, 0),
                                                        Gaussian(-1, 1, 2, 0)]))
    
    
    fig, axs = plt.subplots(ncols=4, nrows=len(h_s), figsize=(4 * 6, len(h_s) * 6))
    for i in range(4):
        for h_seed, h in h_s.items():
            ax = axs[h_seed, i]
            plot_hyperplane_2d(h, ax=ax, lb=-1, ub=1)
            ax.scatter(x=dists[i][:, 0], y=dists[i][:, 1])
            ax.set_title(f"Hyperplane {h_seed}")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
    plt.show()



def plot_hyperplane_2d(H: np.ndarray,
                       ax: plt.Axes,
                       lb: int,
                       ub: int):
    for i, h in enumerate(H):

        a1 = h[0]
        a2 = h[1]
        b = h[-1]
        if a2 != 0:
            x1 = np.linspace(lb, ub, 10)
            x2 = (-a1 * x1 - b) / a2
        else:
            x2 = np.linspace(lb, ub, 10)
            x1 = (-a2 * x2 - b) / a1

        ax.plot(x1, x2, label=f'{i}')


def plot(hyperplanes: np.ndarray,
         faces: np.ndarray,
         regions: List,
         items: np.ndarray,
         lb: int,
         ub: int):

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_hyperplane_2d(hyperplanes, ax, lb, ub)
    plot_hyperplane_2d(faces, ax, lb, ub)

    for i, item in enumerate(items):
        ax.scatter(item[0], item[1])
        ax.text(item[0], item[1], f'x {i}')
    for i, region in enumerate(regions):
        ax.text(region.center[0], region.center[1], f'region {i}')
        # plt.scatter(region.center[0], region.center[1])

    ax.set_xlim([lb, ub])
    ax.set_ylim([lb, ub])
    plt.show()
