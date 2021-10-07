from math import log, ceil, floor
from collections import Counter

import numpy as np


def sum_dicts(d1, d2):
    return {k: d1.get(k, []) + d2.get(k, []) for k in set(d1) | set(d2)}


def mode(a):
    c = Counter(a)
    return c.most_common(1)[0][0]


def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return 2 ** min(possible_results, key=lambda z: abs(x - 2 ** z))


def baseline_results(hps, fn):
    baseline_params = dict()
    baseline_params['server_lr'] = fn(hps["lr"])
    baseline_params['server_momentum'] = fn(hps["momentum"])
    baseline_params['batch_size'] = closest_power(fn(hps["batch_size"]))

    return baseline_params


def filter_outliers(a):
    q1 = np.percentile(a, 25)
    q3 = np.percentile(a, 75)
    iqr = q3 - q1

    min_lim = q1 - 1.5 * iqr
    max_lim = q3 + 1.5 * iqr

    return [x for x in a if min_lim <= x <= max_lim]
