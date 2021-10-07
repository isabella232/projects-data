from utils import closest_power, filter_outliers


def lr_heuristic(ratios, nr_parties, lrs, best_acc, val_accs, type_of_skew, v=0):
    agg_lr = 0
    for i in range(nr_parties):
        if type_of_skew == "feature":
            # abs not in paper
            agg_lr += lrs[i] * (1 - lrs[i]) / (10 * nr_parties) * (1 - abs(best_acc - val_accs[i]))

        if type_of_skew == "label":
            # abs not in paper
            agg_lr += lrs[i] * (1 - lrs[i]) * (1 - abs(best_acc - val_accs[i])) / (nr_parties * 10)

        if type_of_skew == "qty":
            # abs not in paper
            agg_lr += ratios[i] * lrs[i] * (1 - lrs[i]) * (1 - abs(best_acc - val_accs[i])) / 10

    return agg_lr


def momentum_heuristic(ratios, nr_parties, momentums, best_acc, val_accs, type_of_skew, v=0):
    agg_mom = 0
    if type_of_skew == "feature":
        # abs not in paper
        if v == 0:
            for i in range(nr_parties):
                agg_mom += min(1 / nr_parties,
                               momentums[i] * (1 - momentums[i]) / nr_parties * (1 - abs(best_acc - val_accs[i])) * 10)
        elif v == 1:
            fil_moms = filter_outliers(momentums)
            fil_nr_parties = len(fil_moms)
            for mom in fil_moms:
                agg_mom += mom / fil_nr_parties

    elif type_of_skew == "label":
        for i in range(nr_parties):
            agg_mom += momentums[i] / nr_parties

    elif type_of_skew == "qty":
        # abs not in paper
        if v == 0:
            for i in range(nr_parties):
                agg_mom += min(1 / nr_parties,
                               ratios[i] * momentums[i] * (1 - momentums[i]) * (1 - abs(best_acc - val_accs[i])) * 10)
        elif v == 1:
            for i in range(nr_parties):
                # delete min, best_acc, val_acc
                agg_mom += momentums[i] / nr_parties
    return agg_mom


def batch_size_heuristic(ratios, nr_parties, batch_sizes, val_accs, type_of_skew, v=0):
    agg_bs = 0
    if type_of_skew == "feature":
        if v == 0:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] * val_accs[i] / nr_parties

        elif v == 1:
            for i in range(nr_parties):
                agg_bs += batch_sizes[i] / nr_parties

    elif type_of_skew == "label":
        for i in range(nr_parties):
            agg_bs += batch_sizes[i] * val_accs[i] / nr_parties

    elif type_of_skew == "qty":
        for i in range(nr_parties):
            agg_bs += ratios[i] * batch_sizes[i] * val_accs[i]

    return closest_power(agg_bs)


def aggregate_results(hps, accs, best_acc, ratios, type_of_skew, v=0):
    nr_parties = len(ratios)

    agg_params = dict()

    agg_params['server_lr'] = lr_heuristic(ratios=ratios, nr_parties=nr_parties, lrs=hps["lr"],
                                           best_acc=best_acc, val_accs=accs,
                                           type_of_skew=type_of_skew, v=v)
    agg_params['server_momentum'] = momentum_heuristic(ratios=ratios, nr_parties=nr_parties,
                                                       momentums=hps["momentum"], best_acc=best_acc,
                                                       val_accs=accs, type_of_skew=type_of_skew, v=v)
    agg_params['batch_size'] = batch_size_heuristic(ratios=ratios, nr_parties=nr_parties,
                                                    batch_sizes=hps["batch_size"], val_accs=accs,
                                                    type_of_skew=type_of_skew, v=v)

    return agg_params
