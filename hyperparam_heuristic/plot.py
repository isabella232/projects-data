import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import baseline_results, mode
from read_data import get_client_res, get_fedavg_res
from heuristic_funcs import aggregate_results


# TODO: separate aggregate_results from plot results function
def plot_results(dataset, skews, nrs_parties, hyper_param, type_of_skew):
    fig, ax = plt.subplots(nrows=len(skews), ncols=len(nrs_parties), figsize=(12, 20))
    agg_aes = {"v0": [], "v1": []}
    baseline_aes = {"mean": [], "median": [], "mode": []}
    for i, s in enumerate(skews):
        for j, p in enumerate(nrs_parties):
            hps, accs, best_acc, ratios = get_client_res(dataset, s, p, type_of_skew)

            agg_params_v0 = aggregate_results(hps, accs, best_acc, ratios, type_of_skew, v=0)
            agg_params_v1 = aggregate_results(hps, accs, best_acc, ratios, type_of_skew, v=1)
            mean_params = baseline_results(hps, np.mean)
            median_params = baseline_results(hps, np.median)
            if hyper_param == "batch_size":
                mode_params = baseline_results(hps, mode)
            fedavg_params = get_fedavg_res(dataset, s, p, type_of_skew)

            sns.boxplot(y=hps[hyper_param], showmeans=True, ax=ax[i, j])
            sns.swarmplot(y=hps[hyper_param], color=".25", ax=ax[i, j])
            if hyper_param in ["lr", "momentum"]:
                prefix = "server_"
            else:
                prefix = ""

            fedavg_res = fedavg_params[f"{prefix}{hyper_param}"]

            heuristic_res_v0 = agg_params_v0[f"{prefix}{hyper_param}"]
            heuristic_res_v1 = agg_params_v1[f"{prefix}{hyper_param}"]

            mean_res = mean_params[f"{prefix}{hyper_param}"]
            median_res = median_params[f"{prefix}{hyper_param}"]

            if hyper_param == "batch_size":
                mode_res = mode_params[f"{prefix}{hyper_param}"]

            ax[i, j].scatter(x=0, y=heuristic_res_v0, s=100, marker="o", c="red",
                             label="heuristic v0")
            ax[i, j].scatter(x=0, y=heuristic_res_v1, s=100, marker="x", c="orange",
                             label="heuristic v1")
            ax[i, j].scatter(x=0, y=fedavg_res, s=100, marker="o", c="green",
                             label="fedavg")

            agg_aes["v0"].append(np.abs(heuristic_res_v0 - fedavg_res))
            agg_aes["v1"].append(np.abs(heuristic_res_v1 - fedavg_res))
            baseline_aes["mean"].append(np.abs(mean_res - fedavg_res))
            baseline_aes["median"].append(np.abs(median_res - fedavg_res))
            if hyper_param == "batch_size":
                baseline_aes["mode"].append(np.abs(mode_res - fedavg_res))
                ax[i, j].set_yscale("log", base=2)

            ax[i, j].set_title(f"{len(ratios)} clients, {type_of_skew} skew {s}, {dataset}")
            ax[i, j].set_xlabel("Number of clients")
            ax[i, j].set_ylabel(f"{hyper_param}")
            ax[i, j].legend()

    return agg_aes, baseline_aes
