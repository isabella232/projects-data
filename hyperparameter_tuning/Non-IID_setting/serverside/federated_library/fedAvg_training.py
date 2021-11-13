import fedjax
from fedjax import core

from federated_library.models_haiku import get_model
from federated_library.distributions import qty_skew_distrib, label_skew_distrib, feature_skew_distrib, iid_distrib


def train_fedAvg(params, ds, test_split, ds_info, custom_model=None, display=False):
    """ Run training and test with FedJAX FedAvg

    :param params: parameters to train on, dictionary with keys [skew, client_lr, server_lr, client_momentum, server_momentum, batch_size, epochs_per_round, rounds]
    :param ds: train dataset
    :param test_split: test dataset
    :param ds_info: dataset information
    :param custom_model: optional custom model
    :param display: True if distributions graphs are wanted
    :return: test metrics result
    """
    skew_type = ds_info['skew_type']
    seed = ds_info['seed']

    x_train = ds[0]
    y_train = ds[1]

    if skew_type == "qty":
        print("Quantity skew")
        federated_data = qty_skew_distrib(
            x_train, y_train, ds_info, params['skew'], decentralized=False, display=display)
    elif skew_type == "label":
        print("Label skew")
        federated_data = label_skew_distrib(
            x_train, y_train, ds_info, params['skew'], decentralized=False, display=display)
    elif skew_type == "feature":
        print("Feature skew")
        federated_data = feature_skew_distrib(
            x_train, y_train, ds_info, params['skew'], decentralized=False, display=display)
    else:
        print("IID distribution")
        federated_data = iid_distrib(
            x_train, y_train, ds_info, decentralized=False, display=display)

    model = get_model(params, ds_info, custom_model)

    algorithm = fedjax.algorithms.FedAvg(
        federated_data=federated_data,
        model=model,
        client_optimizer=core.get_optimizer(
            core.OptimizerName.MOMENTUM, learning_rate=params['client_lr'], momentum=params['client_momentum']),
        server_optimizer=core.get_optimizer(
            core.OptimizerName.MOMENTUM, learning_rate=params['server_lr'], momentum=params['server_momentum']),
        hparams=fedjax.algorithms.FedAvgHParams(
            train_data_hparams=core.ClientDataHParams(
                batch_size=params['batch_size'],
                num_epochs=params['epochs_per_round'])),
        rng_seq=core.PRNGSequence(seed))

    state = algorithm.init_state()

    for _ in range(params['rounds']):
        state = algorithm.run_round(state, federated_data.client_ids)

    test_metrics = core.evaluate_single_client(
        dataset=test_split, model=model, params=state.params)

    return test_metrics
