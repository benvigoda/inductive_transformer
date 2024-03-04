from flax import linen as nn
import jax.numpy as jnp
from helper_functions import custom_normalize


class EncoderAttentionPi(nn.Module):
    vocab_size: int
    layer_width: int
    hyperparams.

    @nn.compact
    def __call__(self, v):

        assert v.shape == (self.layer_width, self.layer_width)
        # we expect v to be already normalized categorical

        prob_weights = nn.relu(self.weights) + 1e-9

        prob_weights = custom_normalize(prob_weights, dim=1)

        # element-wise product of weight vector and token vector for each column in the layer
        y = prob_weights * v

        # make it an inner product by taking a sum along the choice dimension
        y = jnp.sum(y, axis=0, keepdim=True)  # after summing it is size = (1, layer_width)
        assert y.shape == (1, self.layer_width)

        y = custom_normalize(y, axis=1)

        return y  # y is categorical
