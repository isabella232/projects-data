import numpy as np

from utils import closest_power, filter_outliers
from constants import HEUR_DICT


def lr_heuristic(ratios, nr_parties, hp_values, best_acc, val_accs, type_of_skew, v=0):
    agg_lr = 0
    lrs = hp_values
    if type_of_skew == "feature":
        if v == 0:
            for i in range(nr_parties):
                # abs not in paper
                agg_lr += lrs[i] * (1 - lrs[i]) / (10 * nr_parties) * (1 - abs(best_acc - val_accs[i]))

        elif v == 1:
            for i in range(nr_parties):
                # remove best_acc and abs
                agg_lr += lrs[i] * (1 - lrs[i]) / (10 * nr_parties)

        if v == 2:
            # TODO
            agg_lr = 0.032

    if type_of_skew == "label":
        if v == 0:
            for i in range(nr_parties):
                # abs not in paper
                agg_lr += lrs[i] * (1 - lrs[i]) / (10 * nr_parties) * (1 - abs(best_acc - val_accs[i]))
        elif v == 1:
            for i in range(nr_parties):
                # remove best_acc and abs
                agg_lr += lrs[i] * (1 - lrs[i]) / (10 * nr_parties)

        if v == 2:
            # TODO
            agg_lr = 0.032

    if type_of_skew == "qty":
        if v == 0:
            for i in range(nr_parties):
                # abs not in paper
                agg_lr += ratios[i] * lrs[i] * (1 - lrs[i]) * (1 - abs(best_acc - val_accs[i])) / 10
        elif v == 1:
            for i in range(nr_parties):
                # remove best_acc and abs
                agg_lr += ratios[i] * lrs[i] * (1 - lrs[i]) / 10

        if v == 2:
            # TODO
            agg_lr = .032

    return agg_lr


def momentum_heuristic(ratios, nr_parties, hp_values, best_acc, val_accs, type_of_skew, v=0):
    agg_mom = 0
    momentums = hp_values
    if type_of_skew == "feature":
        # abs not in paper
        if v == 0:
            for i in range(nr_parties):
                agg_mom += min(1 / nr_parties,
                               momentums[i] * (1 - momentums[i]) / nr_parties * (1 - abs(best_acc - val_accs[i])) * 10)
        elif v == 1:
            # CAN'T FIND BETTER HEURISTIC THAN SIMPLE AVERAGE
            agg_mom = np.mean(momentums)

        elif v == 2:
            # SIMPLE AVERAGE FILTERING OUTLIERS
            fil_moms = filter_outliers(momentums)
            agg_mom = np.mean(fil_moms)

    elif type_of_skew == "label":
        if v == 0:
            agg_mom = np.mean(momentums)
        elif v == 1:
            # CAN'T FIND BETTER HEURISTIC THAN SIMPLE AVERAGE
            agg_mom = np.mean(momentums)

        elif v == 2:
            # SIMPLE AVERAGE FILTERING OUTLIERS
            fil_moms = filter_outliers(momentums)
            agg_mom = np.mean(fil_moms)

    elif type_of_skew == "qty":
        if v == 0:
            for i in range(nr_parties):
                # abs not in paper
                agg_mom += min(1 / nr_parties,
                               ratios[i] * momentums[i] * (1 - momentums[i]) * (1 - abs(best_acc - val_accs[i])) * 10)
        elif v == 1:
            # CAN'T FIND BETTER HEURISTIC THAN SIMPLE AVERAGE
            agg_mom = np.mean(momentums)

        elif v == 2:
            # SIMPLE AVERAGE FILTERING OUTLIERS
            fil_moms = filter_outliers(momentums)
            agg_mom = np.mean(fil_moms)

    return agg_mom


def batch_size_heuristic(ratios, nr_parties, hp_values, best_acc, val_accs, type_of_skew, v=0):
    agg_bs = 0
    batch_sizes = hp_values
    if type_of_skew == "feature":
        if v == 0:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] * val_accs[i] / nr_parties

        elif v == 1:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] / nr_parties

        if v == 2:
            # SIMPLE AVERAGE FILTERING OUTLIERS
            fil_bs = filter_outliers(batch_sizes)
            agg_bs += np.mean(fil_bs)

    elif type_of_skew == "label":
        if v == 0:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] * val_accs[i] / nr_parties

        elif v == 1:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] / nr_parties

        if v == 2:
            # SIMPLE AVERAGE FILTERING OUTLIERS
            fil_bs = filter_outliers(batch_sizes)
            agg_bs += np.mean(fil_bs)

    elif type_of_skew == "qty":
        if v == 0:
            for i in range(nr_parties):
                agg_bs += ratios[i] * batch_sizes[i] * val_accs[i]

        elif v == 1:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] / nr_parties

        if v == 2:
            # SIMPLE AVERAGE FILTERING OUTLIERS
            fil_bs = filter_outliers(batch_sizes)
            agg_bs += np.mean(fil_bs)

    return closest_power(agg_bs)


def aggregate_results(hps, accs, best_acc, ratios, type_of_skew, hp_name=None, v=0):
    nr_parties = len(ratios)

    heuristic_fns = dict(zip(HEUR_DICT.keys(), [lr_heuristic, momentum_heuristic, batch_size_heuristic]))

    agg_params = dict()
    if hp_name:
        hp_in = HEUR_DICT[hp_name]["in"]
        hp_out = HEUR_DICT[hp_name]["out"]
        agg_params[hp_out] = heuristic_fns[hp_name](ratios=ratios, nr_parties=nr_parties,
                                                        hp_values=hps[hp_in],
                                                        best_acc=best_acc,
                                                        val_accs=accs,
                                                        type_of_skew=type_of_skew, v=v)

    else:
        for hp, heur_fn in heuristic_fns.items():
            hp_in = HEUR_DICT[hp]["in"]
            hp_out = HEUR_DICT[hp]["out"]
            agg_params[HEUR_DICT[hp]["out"]] = heur_fn(ratios=ratios, nr_parties=nr_parties, hp_values=hps[hp_in],
                                                       best_acc=best_acc, val_accs=accs,
                                                       type_of_skew=type_of_skew, v=v)

    return agg_params
