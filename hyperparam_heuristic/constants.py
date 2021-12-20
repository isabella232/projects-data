TYPE_OF_SKEW = ["feature", "label", "qty"]
DATASETS = ["MNIST", "SVHN_CROPPED", "CIFAR10", "EMNIST"]

FEATURE_SKEWS = ["0.02", "0.1"]
LABEL_SKEWS = ["0.1", "1.0", "5.0"]
QTY_SKEWS = ["0.1", "0.4", "1.0", "2.0"]

SKEWS = {
    "feature": [float(s) for s in FEATURE_SKEWS],
    "label": [float(s) for s in LABEL_SKEWS],
    "qty": [float(s) for s in QTY_SKEWS]
}

NR_PARTIES = [10, 20]

HEUR_DICT = {
    "lr": {
        "in": "client_lr",
        "out": "server_lr",
    },
    "mom": {
        "in": "client_momentum",
        "out": "server_momentum",
    },
    "bs": {
        "in": "batch_size",
        "out": "batch_size",
    }
}

INPUT_HEUR = [HEUR_DICT[hp]["in"] for hp in HEUR_DICT.keys()]
OUTPUT_HEUR = [HEUR_DICT[hp]["out"] for hp in HEUR_DICT.keys()]
