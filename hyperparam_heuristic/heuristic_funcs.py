import numpy as np

from utils import closest_power, filter_outliers
from constants import HEUR_DICT


def lr_heuristic(ratios, nr_parties, hp_values, best_acc, val_accs, type_of_skew, v=0):
    agg_lr = 0
    lrs = hp_values
    if type_of_skew == "feature":
        if v == 0:
            for i in range(nr_parties):
                agg_lr += lrs[i] * (1 - lrs[i]) / (10 * nr_parties)

            # add bias
            agg_lr += .482

        elif v == 1:
            # constant
            agg_lr = 0.5

    if type_of_skew == "label":
        if v == 0:
            # linear
            for i in range(nr_parties):
                agg_lr += lrs[i] * (1 - lrs[i]) / \
                    (10 * nr_parties)
            agg_lr += .483 / nr_parties
        elif v == 1:
            # non-linear
            agg_lr = max(lrs)
        elif v == 2:
            # constant
            agg_lr = 0.5

    if type_of_skew == "qty":
        if v == 0:
            for i in range(nr_parties):
                agg_lr += (ratios[i] * lrs[i] *
                           val_accs[i]) / nr_parties

            agg_lr += .432

        if v == 1:
            # constant
            agg_lr = .456

    return agg_lr


def momentum_heuristic(ratios, nr_parties, hp_values, best_acc, val_accs, type_of_skew, v=0):
    agg_mom = 0
    moms = hp_values
    if type_of_skew == "feature":
        # abs not in paper
        if v == 0:
            for i in range(nr_parties):
                agg_mom += moms[i] * (1 - moms[i]) / nr_parties

            agg_mom += .627
            # SIMPLE AVERAGE FILTERING OUTLIERS
            # fil_moms = filter_outliers(moms)
            # agg_mom = np.mean(fil_moms)

        elif v == 1:
            agg_mom = .712

    elif type_of_skew == "label":
        if v == 0:
            agg_mom = np.mean(moms)
        elif v == 1:
            # fil_moms = filter_outliers(moms)
            # nr_parties = len(fil_moms)

            for i in range(nr_parties):
                agg_mom += (ratios[i] * moms[i] * val_accs[i]) / nr_parties

        elif v == 2:
            agg_mom = .763

    elif type_of_skew == "qty":
        if v == 0:
            for i in range(nr_parties):
                # abs not in paper
                agg_mom += min(1 / nr_parties,
                               ratios[i] * moms[i] * (1 - moms[i]) * (1 - abs(best_acc - val_accs[i])) * 10)
        elif v == 1:
            for i in range(nr_parties):
                agg_mom += (ratios[i] * moms[i] * val_accs[i]) / nr_parties
            # # SIMPLE AVERAGE FILTERING OUTLIERS
            # fil_moms = filter_outliers(moms)
            # agg_mom = np.mean(fil_moms)

        elif v == 2:
            agg_mom = .412

    return agg_mom


def batch_size_heuristic(ratios, nr_parties, hp_values, best_acc, val_accs, type_of_skew, v=0):
    agg_bs = 0
    batch_sizes = hp_values
    if type_of_skew == "feature":
        if v == 0:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] * val_accs[i] / nr_parties

        elif v == 1:
            agg_bs = 8

    elif type_of_skew == "label":
        if v == 0:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] * val_accs[i] / nr_parties

        if v == 1:
            # SIMPLE AVERAGE FILTERING OUTLIERS
            fil_bs = filter_outliers(batch_sizes)
            agg_bs += np.mean(fil_bs)

    elif type_of_skew == "qty":
        if v == 0:
            for i in range(nr_parties):
                agg_bs += ratios[i] * batch_sizes[i] * val_accs[i]

        if v == 1:
            # SIMPLE AVERAGE FILTERING OUTLIERS
            fil_bs = filter_outliers(batch_sizes)
            agg_bs += np.mean(fil_bs)

    return closest_power(agg_bs)


def aggregate_results(hps, accs, best_acc, ratios, type_of_skew, hp_name=None, v=0):
    nr_parties = len(ratios)

    heuristic_fns = dict(
        zip(HEUR_DICT.keys(), [lr_heuristic, momentum_heuristic, batch_size_heuristic]))

    agg_params = dict()
    if hp_name:
        hp_in = HEUR_DICT[hp_name]["in"]
        hp_out = HEUR_DICT[hp_name]["out"]
        agg_params[hp_out] = \
            heuristic_fns[hp_name](ratios=ratios, nr_parties=nr_parties,
                                   hp_values=hps[hp_in],
                                   best_acc=best_acc,
                                   val_accs=accs,
                                   type_of_skew=type_of_skew, v=v)

    else:
        for hp, heur_fn in heuristic_fns.items():
            hp_in = HEUR_DICT[hp]["in"]
            hp_out = HEUR_DICT[hp]["out"]
            agg_params[HEUR_DICT[hp]["out"]] = \
                heur_fn(ratios=ratios, nr_parties=nr_parties, hp_values=hps[hp_in],
                        best_acc=best_acc, val_accs=accs,
                        type_of_skew=type_of_skew, v=v)

    return agg_params
