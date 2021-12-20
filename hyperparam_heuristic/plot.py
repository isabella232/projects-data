from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import baseline_results, mode
from read_data import get_client_res, get_fedavg_res
from heuristic_funcs import aggregate_results
from constants import HEUR_DICT


# TODO: separate aggregate_results from plot results function
def plot_results(dataset, skews, nrs_parties, type_of_skew, hp_name, versions=(0, 1)):
    fig, ax = plt.subplots(nrows=len(skews), ncols=len(
        nrs_parties), figsize=(10, 20), sharey=True)
    agg_errors = defaultdict(list)
    baseline_es = defaultdict(list)
    fedavg_results = []

    hp_in = HEUR_DICT[hp_name]["in"]
    hp_out = HEUR_DICT[hp_name]["out"]

    for i, s in enumerate(skews):
        for j, p in enumerate(nrs_parties):
            hps, accs, best_acc, ratios = get_client_res(dataset, s, p, type_of_skew,
                                                         hp_name=hp_in)

            agg_params = [
                aggregate_results(
                    hps, accs, best_acc, ratios, type_of_skew, hp_name, v=v
                )
                for v in versions
            ]
            mean_params = baseline_results(hps, np.mean, hp_name)
            median_params = baseline_results(hps, np.median, hp_name)
            if hp_name == "bs":
                mode_params = baseline_results(hps, mode, hp_name)

            sns.boxplot(y=hps[hp_in], showmeans=True, ax=ax[i, j])
            sns.swarmplot(y=hps[hp_in], color=".25", ax=ax[i, j])
            try:
                fedavg_params = get_fedavg_res(dataset, s, p, type_of_skew)
            except FileNotFoundError as e:
                break
            fedavg_res = fedavg_params[hp_out]

            fedavg_results.append((hps[hp_in], fedavg_res))

            heuristic_res = [agg_params[v][hp_out] for v in versions]

            mean_res = mean_params[hp_out]
            median_res = median_params[hp_out]

            if hp_name == "bs":
                mode_res = mode_params[hp_out]

            for v in versions:
                ax[i, j].scatter(x=0, y=heuristic_res[v], s=100,
                                 label=f"heuristic v{v}")
            ax[i, j].scatter(x=0, y=fedavg_res, s=100, marker="x",
                             label="fedavg")

            for v in versions:
                agg_errors[v].append(heuristic_res[v] - fedavg_res)

            baseline_es["mean"].append(mean_res - fedavg_res)
            baseline_es["median"].append(median_res - fedavg_res)
            if hp_name == "bs":
                baseline_es["mode"].append(mode_res - fedavg_res)
                ax[i, j].set_yscale("log", base=2)

            ax[i, j].set_title(
                f"{len(ratios)} clients, {type_of_skew} skew {s}, {dataset}")
            ax[i, j].set_xlabel("Number of clients")
            ax[i, j].set_ylabel(hp_name)
            ax[i, j].legend()

    return agg_errors, baseline_es, fedavg_results


def print_results(heuristic_es, baseline_es, results, versions=(0, 1)):
    for res in results:
        print(res)
    fedavg_res = [f[1] for f in results]
    # print(f"fedavg server hp results: {fedavg_res}")
    print(
        f"(mean: {np.mean(fedavg_res):.3f}, variance: {np.var(fedavg_res):.3f})")

    print("\nHEURISTIC RESULTS")

    for v in versions:
        print(
            f"Heuristic {v} MAE: {mae(heuristic_es[v]):.3f}"
            f" ± {std(heuristic_es[v]):.3f}"
            f" (Bias: {np.mean(heuristic_es[v]):+.3f})")

    print(
        f"Mean MAE: {mae(baseline_es['mean']):.3f}"
        f" ± {std(baseline_es['mean']):.3f}"
        f" (Bias: {np.mean(baseline_es['mean']):+.3f})")
    print(
        f"Median MAE: {mae(baseline_es['median']):.3f}"
        f" ± {std(baseline_es['median']):.3f}"
        f" (Bias: {np.mean(baseline_es['median']):+.3f})")


def mae(es):
    return np.mean(np.abs(es))


def std(es):
    return np.std(np.abs(es))
