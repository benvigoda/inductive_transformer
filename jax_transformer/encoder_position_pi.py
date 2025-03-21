from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore
from jax_transformer.helper_functions import (
    custom_normalize,
    EPSILON,
)


"""
Let vocab_size = 4, num_positions = 3, and layer_width = 2

The data is then a tensor that is size (num_positions=3, vocab size=4)

There is a Forney equals gates at each specific (position and word)

The left column has three pi_t's, each with 4 vocab words that can explain away the entire data
The right column also has this.

When the data says "small dog", then we want the left column to explain away the data

When the data says "big cat", then we want the right column to explain away the data

The entire layer_width must participate in explaining away the changing data.

We will need an open closed universe

... maybe we can do this in a simpler way that will not require the open closed universe:
in encoder_layer.py clone the data and send one clone straight up and one clone across
"""


class EncoderPositionPi(nn.Module):
    num_positions: int
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, rho):
        assert rho.shape == (self.num_positions, self.layer_width)
        # we need to normalize rho
        # rho = custom_normalize(rho, dim=0)

        weights = self.param(
            "weights", self.weight_init, (self.num_positions, self.layer_width)
        )
        prob_weights = nn.relu(weights) + EPSILON
        # NOTE: we decided to normalize the weights (it shouldn't matter)
        prob_weights = custom_normalize(prob_weights, axis=0)

        # Add this?
        # rho = custom_normalize(rho, axis=0)

        # element-wise product of weight vector and token vector for each column in the layer
        x = prob_weights * rho

        # make it an inner product by taking a sum along the token dimension
        x = jnp.sum(x, axis=0, keepdims=True)
        assert x.shape == (1, self.layer_width)
        # x = custom_normalize(x, axis=1)
        return x  # x is categorical
