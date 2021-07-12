from haiku_dropout import Dropout
import collections
import numpy as np
import haiku as hk
from metrics import *


def get_model(params, ds_info, custom_model=None):
    """ Get default haiku model for mnist, emnist, svhn or cifar10
        or a custom haiku model

    :param params: parameters dict for the model, especially activation function 'act_fn'
    :param ds_info: dataset information
    :param custom_model: an optional custom haiku model (default is None)
    :return: haiku model
    """

    dataset_name = ds_info['dataset_name']
    num_classes = ds_info['num_classes']
    sample_shape = ds_info['sample_shape']
    seed = ds_info['seed']

    sample_height, sample_width, sample_channels = sample_shape

    # Defines the expected structure of input batches to the model. This is used to
    # determine the model parameter shapes.
    input_shape = collections.OrderedDict(
        x=np.zeros((1, sample_height, sample_width, sample_channels)),
        y=np.zeros(1, ))

    metrics = collections.OrderedDict(accuracy=accuracy)

    ## PRECISION AND RECALL COMING LATER
    ## ideas here : https://github.com/poets-ai/elegy/tree/master/elegy/metrics

    def forward_pass(batch):
        model_emnist_svhn = hk.Sequential([
            hk.Conv2D(output_channels=6, kernel_shape=5, padding='VALID'),
            params['act_fn'],
            hk.AvgPool(window_shape=2, strides=1, padding='VALID'),
            hk.Conv2D(output_channels=16, kernel_shape=5, padding='VALID'),
            params['act_fn'],
            hk.AvgPool(window_shape=2, strides=1, padding='VALID'),
            hk.Flatten(),
            hk.Linear(120),
            params['act_fn'],
            hk.Linear(84),
            params['act_fn'],
            hk.Linear(num_classes)
        ])

        model_mnist = hk.Sequential([
            hk.Conv2D(output_channels=32, kernel_shape=5, padding='VALID'),
            params['act_fn'],
            hk.AvgPool(window_shape=3, strides=1, padding='VALID'),
            hk.Conv2D(output_channels=32, kernel_shape=5, padding='VALID'),
            params['act_fn'],
            hk.AvgPool(window_shape=2, strides=1, padding='VALID'),
            hk.Flatten(),
            hk.Linear(200),
            params['act_fn'],
            hk.Linear(num_classes)
        ])

        model_cifar10 = hk.Sequential([
            hk.Conv2D(output_channels=64, kernel_shape=3, padding='VALID'),
            params['act_fn'],
            hk.Conv2D(output_channels=64, kernel_shape=3, padding='VALID'),
            params['act_fn'],
            hk.AvgPool(window_shape=2, strides=1, padding='VALID'),
            Dropout(0.2, seed),
            hk.Conv2D(output_channels=96, kernel_shape=3, padding='VALID'),
            params['act_fn'],
            hk.Conv2D(output_channels=96, kernel_shape=3, padding='VALID'),
            params['act_fn'],
            hk.AvgPool(window_shape=2, strides=1, padding='VALID'),
            Dropout(0.3, seed),
            hk.Conv2D(output_channels=128, kernel_shape=3, padding='VALID'),
            params['act_fn'],
            hk.Conv2D(output_channels=128, kernel_shape=3, padding='VALID'),
            params['act_fn'],
            hk.AvgPool(window_shape=2, strides=1, padding='VALID'),
            Dropout(0.4, seed),
            hk.Flatten(),
            hk.Linear(128),
            params['act_fn'],
            Dropout(0.5, seed),
            hk.Linear(num_classes)
        ])

        if custom_model is None:
            if dataset_name == "mnist":
                return model_mnist(batch['x'])
            elif dataset_name == "emnist" or dataset_name == "svhn_cropped":
                return model_emnist_svhn(batch['x'])
            elif dataset_name == "cifar10":
                return model_cifar10(batch['x'])
        else:
            return custom_model(batch['x'])

    transformed_forward_pass = hk.transform(forward_pass)

    return core.create_model_from_haiku(
        transformed_forward_pass=transformed_forward_pass,
        sample_batch=input_shape,
        loss_fn=mse,
        metrics_fn_map=metrics)
