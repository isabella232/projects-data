import jax
from jax import lax
from typing import Optional
import haiku as hk
import jax.numpy as jnp


class Dropout(hk.Module):
    """
    Custom haiku Dropout class since dropout is implemented in a weird way in haiku
    """

    def __init__(
            self,
            rate=float,
            seed=int,
            training: Optional[bool] = True,
            name: Optional[str] = None,
    ):

        super().__init__(name=name)

        if rate < 0.0 or rate >= 1.0:
            raise ValueError("rate must be in [0, 1)")

        self.rate = rate
        self.seed = seed
        self.training = training

    def __call__(
            self,
            inputs: jnp.ndarray,
            *,
            precision: Optional[lax.Precision] = None,
    ):
        if not inputs.shape:
            raise ValueError("Input must not be scalar")

        if self.training or self.rate == 0.0:
            keep_rate = 1.0 - self.rate
            keep = jax.random.bernoulli(jax.random.PRNGKey(self.seed), keep_rate, shape=inputs.shape)
            return keep * inputs / keep_rate
        else:
            return inputs
