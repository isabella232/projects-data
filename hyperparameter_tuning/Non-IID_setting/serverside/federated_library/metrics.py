from fedjax import core
import jax
import jax.numpy as jnp


# METRICS
def accuracy(batch: core.Batch, preds: jnp.ndarray) -> core.Metric:
    """Compute accuracy

    :param batch: targets
    :param preds: predictions
    :return: accuracy of the predictions against the targets
    """
    return core.metrics.accuracy_fn(targets=batch['y'], preds=preds)


# MSE
def mse(batch: core.Batch, preds: jnp.ndarray) -> core.Metric:
    """Compute softmax of predicitons and compute mean squared error

    :param batch: targets
    :param preds: predictions
    :return: mse between predictions and targets
    """
    num_classes = preds.shape[-1]
    prob_preds = jax.nn.softmax(preds)  # log_softmax gives better performance, but cannot use with HE
    one_hot_targets = jax.nn.one_hot(batch['y'], num_classes)
    err = one_hot_targets - prob_preds
    return core.MeanMetric.from_values(jnp.sum(jnp.square(err), axis=-1))
