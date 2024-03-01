from flax import linen as nn
from typing import Callable
import jax.numpy as jnp

from helper_functions import custom_normalize


class DecoderPositionPi(nn.Module):
    num_positions: int
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, x):
        # we expect x to be already normalized categorical
        assert x.shape == (1, self.layer_width)

        weights = self.param('weights', self.weight_init, (self.num_positions, self.layer_width))
        prob_weights = nn.relu(weights) + 1e-9

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the x value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        prob_weights = custom_normalize(prob_weights, axis=0)  # FIXME: could be causing problems

        x = custom_normalize(x, axis=1)

        # element-wise product of weight tensor and y
        rho = prob_weights * x
        assert rho.shape == (self.num_positions, self.layer_width)
        return rho
