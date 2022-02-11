from collections import defaultdict
import json

import numpy as np
import pandas as pd

from constants import INPUT_HEUR, DATASETS

RESULTS_PATH = "../non_IID_setting/results_aggregation/non-IID_res"


def get_client_res(dataset, skew, nr_parties, type_of_skew, hp_name=None):
    experiment_directory = (f"{RESULTS_PATH}/{type_of_skew}_skew/"
                            f"{dataset.upper()}_non-IID_{type_of_skew}_skew/"
                            f"{nr_parties}_parties")

    if type_of_skew == "qty":
        distributions = []
        with open(
                f"{experiment_directory}/individual/{skew}/{dataset}_qty_skew_"
                f"{skew}_{nr_parties}clients_distribution.txt", "r") as reader:
            for i in range(nr_parties):
                distributions.append(reader.readline())
        ratios = []
    else:
        distributions = None
        ratios = []

    accuracies = []

    best_hp = defaultdict(list)

    # Get individual data
    for i in range(nr_parties):
        file_path = (f"{experiment_directory}/individual/{skew}/{skew}_"
                     f"{type_of_skew}_{nr_parties}clts_clt{i}.txt")
        try:
            client_result = pd.read_csv(file_path)
            if type_of_skew == "qty":
                ratio = float(distributions[i].split(
                    ',')[2].replace(' percentage :', ''))
                ratios.append(ratio)
            else:
                ratios.append(1)
            if hp_name:
                best_hp[hp_name].append(client_result.head(
                    1).get([hp_name]).values[0][0])
            else:
                for hp in INPUT_HEUR:
                    best_hp[hp].append(
                        np.array(client_result.head(1).get([hp]).values[0][0]))

            accuracies.append(client_result.head(
                1).get(['val_accuracy']).values[0][0])

        except FileNotFoundError as e:
            print(
                f"File for client {i} in {dataset} ({skew}, {nr_parties}, "
                f"{type_of_skew}) does not exist.")
            # print(file_path)
            # raise e

    best_acc = np.max(accuracies)

    if type_of_skew != "qty":
        arr_ratios = np.array(ratios) / len(ratios)
    else:
        arr_ratios = np.array(ratios)

    return best_hp, accuracies, best_acc, arr_ratios


def get_fedavg_res(dataset, skew, nr_parties, type_of_skew):
    experiment_directory = f"{RESULTS_PATH}/fed_grid_search_results"

    # Get FEDAVG data
    with open(f"{experiment_directory}/{dataset.lower()}_{type_of_skew}_skew_"
              f"{skew}_{nr_parties}clients.txt", "r") as reader:
        line = reader.readline()

        while not line.startswith("\'client_lr\':"):
            line = line[1:]
        fedavg_data = eval("{" + line[:-2])

    return fedavg_data


def get_fedavg_acc(dataset, skew, nr_parties, type_of_skew):
    experiment_directory = f"{RESULTS_PATH}/fed_grid_search_results"

    # Get FEDAVG data
    with open(f"{experiment_directory}/{dataset.lower()}_{type_of_skew}_skew_"
              f"{skew}_{nr_parties}clients.txt", "r") as reader:
        line = reader.readline()
        fedavg_data = eval(line)
    return fedavg_data[0]


def load_heur_results(v, melt=True):
    heur_results = {}

    for dataset in DATASETS:
        with open(f'heur_results/heur_{dataset}.json') as f:
            heur_results[dataset] = json.load(f)

    df = pd.concat(objs=(
        pd.DataFrame.from_records(heur_results["mnist"][f"{v}"]),
        pd.DataFrame.from_records(heur_results["emnist"][f"{v}"]),
        pd.DataFrame.from_records(heur_results["svhn_cropped"][f"{v}"]),
        pd.DataFrame.from_records(heur_results["cifar10"][f"{v}"]),
    ),
        ignore_index=True
    )

    df.rename(columns={"fedavg_acc": "grid_search_acc"}, inplace=True)

    if melt:
        df = pd.melt(df, id_vars=set(df.columns).difference(["grid_search_acc", "heur_acc"]),
                     value_name="acc", var_name="res_type")

    return df