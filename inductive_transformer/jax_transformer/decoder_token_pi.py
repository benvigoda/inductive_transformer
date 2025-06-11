from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore

from inductive_transformer.jax_transformer.helper_functions import (
    custom_normalize,
    EPSILON,
)


class DecoderTokenPi(nn.Module):
    num_positions: int
    vocab_size: int
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, rho):
        # we expect rho to be already normalized categorical
        assert rho.shape == (self.num_positions, self.layer_width)

        weights = self.param(
            "weights",
            self.weight_init,
            (self.num_positions, self.vocab_size, self.layer_width),
        )
        prob_weights = nn.relu(weights) + EPSILON

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the x value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        # prob_weights = nn.functional.normalize(prob_weights, p=1, dim=0)
        prob_weights = custom_normalize(prob_weights, axis=1)

        # rho = custom_normalize(rho, axis=1)  #FIXME: do we want this? We already do it in the decoder_position_pi
        # element-wise product of weight tensor and rho
        t = prob_weights * rho.reshape((self.num_positions, 1, self.layer_width))
        assert t.shape == (self.num_positions, self.vocab_size, self.layer_width)
        # t = custom_normalize(t, axis=1)
        return t
