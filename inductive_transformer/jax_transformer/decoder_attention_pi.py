from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore
from inductive_transformer.jax_transformer.helper_functions import custom_normalize, EPSILON


class DecoderAttentionPi(nn.Module):
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, y):
        # we expect y to be already normalized categorical
        assert y.shape == (1, self.layer_width)

        weights = self.param('weights', self.weight_init, (self.layer_width, self.layer_width))

        # We want to interpret the weights as probabilities. To ensure they're all strictly between
        # 0 and 1, we pass them through a relu and then normalize.
        prob_weights = nn.relu(weights) + EPSILON

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the y value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        prob_weights = custom_normalize(prob_weights, axis=1)

        # and then since y comes in as categorical of size (1, layer_width)
        y = custom_normalize(y, axis=1)

        # element-wise product of weight tensor and y
        v = prob_weights * y
        assert v.shape == (self.layer_width, self.layer_width)
        return v
