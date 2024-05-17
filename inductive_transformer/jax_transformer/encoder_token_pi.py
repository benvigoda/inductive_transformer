from flax import linen as nn  # type: ignore
import jax.numpy as jnp  # type: ignore
from typing import Callable
from inductive_transformer.jax_transformer.helper_functions import EPSILON


class EncoderTokenPi(nn.Module):
    num_positions: int
    layer_width: int
    vocab_size: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, t):
        assert t.shape == (self.num_positions, self.vocab_size, self.layer_width)
        # we expect t to be already normalized

        weights = self.param('weights', self.weight_init, (self.num_positions, self.vocab_size, self.layer_width))
        prob_weights = nn.relu(weights) + EPSILON
        # NOTE: we decided not to normalize the weights (it shouldn't matter)
        # prob_weights = nn.functional.normalize(prob_weights, p=1, dim=0)
        # prob_weights = custom_normalize(prob_weights, dim=1)

        # element-wise product of weight vector and token vector for each column in the layer
        rho = prob_weights * t

        # make it an inner product by taking a sum along the token dimension
        rho = jnp.sum(rho, axis=1)  # after summing it is size = (num_positions, layer_width)
        # rho = custom_normalize(rho, dim=1)

        return rho  # rho is categorical
