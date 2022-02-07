from math import log, ceil, floor
from collections import Counter

import numpy as np

from constants import HEUR_DICT, INPUT_HEUR, OUTPUT_HEUR


def sum_dicts(d1, d2):
    return {k: d1.get(k, []) + d2.get(k, []) for k in set(d1) | set(d2)}


def mode(a):
    c = Counter(a)
    return c.most_common(1)[0][0]


def closest_power(x, pow=2):
    possible_results = floor(log(x, pow)), ceil(log(x, pow))
    return pow ** min(possible_results, key=lambda z: abs(x - pow ** z))


def closest(x, lst):
    closest_idx = np.abs(x - np.asarray(lst)).argmin()
    return lst[closest_idx]


def baseline_results(hps, fn, hyper_param=None):
    baseline_params = dict()
    if hyper_param:
        hp_in = HEUR_DICT[hyper_param]["in"]
        hp_out = HEUR_DICT[hyper_param]["out"]
        baseline_params[hp_out] = fn(hps[hp_in])
    else:
        for hp_in, hp_out in zip(INPUT_HEUR, OUTPUT_HEUR):
            baseline_params[hp_out] = fn(hps[hp_in])
    return baseline_params


def filter_outliers(a, scale=1.5):
    q1 = np.percentile(a, 25)
    q3 = np.percentile(a, 75)
    iqr = q3 - q1

    min_lim = q1 - scale * iqr
    max_lim = q3 + scale * iqr

    return [x for x in a if min_lim <= x <= max_lim]


def compute_accuracy(y, y_hat, grid):
    N = len(y_hat)
    y_hat_proj = np.zeros_like(y_hat)

    for i in range(len(y_hat)):
        for j in range(len(y_hat[0])):
            y_hat_proj[i, j] = closest(y_hat[i][j], grid)

    accs = np.sum((y_hat_proj.T - y) == 0, axis=1) / N

    return accs
