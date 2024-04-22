from flax import linen as nn  # type: ignore
import jax.numpy as jnp  # type: ignore
from typing import Callable
from jax_transformer.helper_functions import custom_normalize


class EncoderAttentionPi(nn.Module):
    vocab_size: int
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, v):

        assert v.shape == (self.layer_width, self.layer_width)
        # we expect v to be already normalized categorical
        weights = self.param('weights', self.weight_init, (self.layer_width, self.layer_width))
        prob_weights = nn.relu(weights) + 1e-9

        prob_weights = custom_normalize(prob_weights, axis=1)

        # element-wise product of weight vector and token vector for each column in the layer
        y = prob_weights * v

        # make it an inner product by taking a sum along the choice dimension
        y = jnp.sum(y, axis=0, keepdims=True)  # after summing it is size = (1, layer_width)
        assert y.shape == (1, self.layer_width)

        y = custom_normalize(y, axis=1)

        return y  # y is categorical
