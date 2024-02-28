import pickle
import pandas as pd
from multiprocessing.pool import Pool
from glob import glob
from src.simulation.result import Result, LuResult


def obtain_result(file):
    with open(file, "rb") as fp:
        result = pickle.load(fp)
        K = -1
        if isinstance(result, LuResult):
            K = result.K
    return {
        "K": K,
        "d": result.d,
        "n_h": result.n_h,
        "user_dist": result.user_dist,
        "n_p": result.n_p,
        "noise": result.noise,
        "method": result.method,
        "tv_err_q_hat": result.tv_err_q_hat,
        "wa_err_q_hat": result.wa_err_q_hat,
    }

def main():
    files = glob("results/*.result")
    with Pool() as p:
        results = p.map(obtain_result, files)
        df = pd.DataFrame(results)
        df.to_parquet("2d_varying_n_p_results.parquet")


if __name__ == "__main__":
    main()

