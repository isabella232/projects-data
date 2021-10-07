import numpy as np
import pandas as pd

PATH = "../hyperparameter_tuning/Non-IID_setting/results_aggregation/non-IID_res"


def get_client_res(dataset, skew, nr_parties, type_of_skew):
    experiment_directory = f"{PATH}/{type_of_skew}_skew/{dataset}_non-IID_{type_of_skew}_skew/{nr_parties}_parties"

    if type_of_skew == "qty":
        distributions = []
        with open(
                f"{experiment_directory}/individual/{skew}/{dataset.lower()}_qty_skew_{skew}_{nr_parties}"
                f"clients_distribution.txt",
                "r") as reader:
            for i in range(nr_parties):
                distributions.append(reader.readline())
        ratios = []
    else:
        distributions = None
        ratios = []
    lrs = []
    momentums = []
    batch_sizes = []
    accuracies = []

    # Get individual data
    for i in range(nr_parties):
        file_path = f"{experiment_directory}/individual/{skew}/{skew}_{type_of_skew}_{nr_parties}clts_clt{i}.txt"
        try:
            client_result = pd.read_csv(file_path)
            if type_of_skew == "qty":
                ratio = float(distributions[i].split(',')[2].replace(' percentage :', ''))
                ratios.append(ratio)
            else:
                ratios.append(1)

            client_lr = client_result.head(1).get(['client_lr']).values[0][0]
            client_momentum = client_result.head(1).get(['client_momentum']).values[0][0]
            client_batch_size = client_result.head(1).get(['batch_size']).values[0][0]
            client_accuracy = client_result.head(1).get(['val_accuracy']).values[0][0]

            lrs.append(client_lr)
            momentums.append(client_momentum)
            batch_sizes.append(client_batch_size)
            accuracies.append(client_accuracy)
        except FileNotFoundError:
            print(f"File for client {i} in {dataset} {(skew, nr_parties, type_of_skew)} does not exist.")
            # print(file_path)

    best_acc = np.max(accuracies)

    if type_of_skew != "qty":
        arr_ratios = np.array(ratios) / len(ratios)
    else:
        arr_ratios = np.array(ratios)

    return {"lr": np.array(lrs),
            "momentum": np.array(momentums),
            "batch_size": np.array(batch_sizes)}, accuracies, best_acc, arr_ratios


def get_fedavg_res(dataset, skew, nr_parties, type_of_skew):
    experiment_directory = f"{PATH}/{type_of_skew}_skew/{dataset}_non-IID_{type_of_skew}_skew/{nr_parties}_parties"
    # Get FEDAVG data
    with open(f"{experiment_directory}/fedavg/{dataset.lower()}_{type_of_skew}_skew_{skew}_{nr_parties}clients.txt",
              "r") as reader:
        line = reader.readline()

        while not line.startswith("\'client_lr\':"):
            line = line[1:]
        fedavg_data = eval("{" + line[:-2])

    return fedavg_data
