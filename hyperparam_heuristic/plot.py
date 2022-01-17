from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import baseline_results, mode, compute_accuracy
from read_data import get_client_res, get_fedavg_res
from heuristic_funcs import aggregate_results
from constants import HEUR_DICT, HP_GRID


# TODO: separate aggregate_results from plot results function
def get_dataset_results(dataset, skews, nrs_parties, type_of_skew, hp_name, versions=(0, 1), plot=True):
    agg_errors = defaultdict(list)
    X = []
    y = []
    y_hat = []

    hp_in = HEUR_DICT[hp_name]["in"]
    hp_out = HEUR_DICT[hp_name]["out"]

    if plot:
        fig, ax = plt.subplots(nrows=len(skews), ncols=len(
            nrs_parties), figsize=(10, 20), sharey=True)

    for i, s in enumerate(skews):
        for j, p in enumerate(nrs_parties):
            # Get results of individual clients
            hps, accs, best_acc, ratios = get_client_res(
                dataset, s, p, type_of_skew, hp_name=hp_in
            )

            # Calculate global hyperparameter using heuristics
            heuristic_res = [
                aggregate_results(
                    hps, accs, best_acc, ratios, type_of_skew, hp_name, v=v
                )[hp_out]
                for v in versions
            ]

            # Baselines
            mean_params = baseline_results(hps, np.mean, hp_name)
            median_params = baseline_results(hps, np.median, hp_name)
            mode_params = baseline_results(hps, mode, hp_name)

            mean_res = mean_params[hp_out]
            median_res = median_params[hp_out]
            mode_res = mode_params[hp_out]

            # Get FEDAVG results (ground truth)
            fedavg_params = get_fedavg_res(dataset, s, p, type_of_skew)
            fedavg_res = fedavg_params[hp_out]

            y_hat.append(heuristic_res)
            y.append(fedavg_res)
            X.append(hps[hp_in])

            for v in versions:
                agg_errors[v].append(heuristic_res[v] - fedavg_res)

            # Calculate errors
            agg_errors["mean"].append(mean_res - fedavg_res)
            agg_errors["median"].append(median_res - fedavg_res)
            agg_errors["mode"].append(mode_res - fedavg_res)

            # Boxplots
            if plot:
                sns.boxplot(y=hps[hp_in], showmeans=False, ax=ax[i, j])
                sns.swarmplot(y=hps[hp_in], color=".25", ax=ax[i, j])

                for v in versions:
                    ax[i, j].scatter(x=0, y=heuristic_res[v], s=100,
                                     label=f"heuristic v{v}")
                ax[i, j].scatter(x=0, y=fedavg_res, s=100, marker="x",
                                 color="yellow", label="fedavg")

                if hp_name == "bs":
                    ax[i, j].set_yscale("log", base=2)

                ax[i, j].set_title(
                    f"{len(ratios)} clients, {type_of_skew} skew {s}, {dataset}")
                ax[i, j].set_xlabel("Number of clients")
                ax[i, j].set_ylabel(hp_name)
                ax[i, j].legend()

    return agg_errors, (X, y, y_hat)


def print_results(agg_es, X, y, y_hat, hp_name, detailed=False):
    print("HEURISTIC RESULTS")

    accs = compute_accuracy(y, y_hat, HP_GRID[hp_name])

    for v in range(len(y_hat[0])):
        print(
            f"Heuristic {v} MAE: {mae(agg_es[v]):.3f}"
            f" ± {std(agg_es[v]):.3f}"
            f" (Bias: {np.mean(agg_es[v]):+.3f})")
        print(f"Heuristic {v} accuracy: {accs[v]}")

    print("BASELINE RESULTS")

    print(
        f"Mean MAE: {mae(agg_es['mean']):.3f}"
        f" ± {std(agg_es['mean']):.3f}"
        f" (Bias: {np.mean(agg_es['mean']):+.3f})")
    print(
        f"Median MAE: {mae(agg_es['median']):.3f}"
        f" ± {std(agg_es['median']):.3f}"
        f" (Bias: {np.mean(agg_es['median']):+.3f})")
    print(
        f"Mode MAE: {mae(agg_es['mode']):.3f}"
        f" ± {std(agg_es['mode']):.3f}"
        f" (Bias: {np.mean(agg_es['mode']):+.3f})")

    if detailed:
        print()
        for i in range(len(X)):
            print("X", X[i])
            print("y", y[i])
            print("y_hat", y_hat[i])

    print(
        f"(mean: {np.mean(y):.3f}, variance: {np.var(y):.3f})")


def mae(es):
    return np.mean(np.abs(es))


def std(es):
    return np.std(np.abs(es))
