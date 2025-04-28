from flax import linen as nn  # type: ignore
import jax.numpy as jnp  # type: ignore
from typing import Callable
from jax_transformer.helper_functions import EPSILON
import jax.numpy as jnp
from jax.nn import logsumexp, log_softmax



class EncoderTokenPi(nn.Module):
    num_positions: int
    layer_width: int
    vocab_size: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, t):
        assert t.shape == (self.num_positions, self.vocab_size, self.layer_width)
        # we expect t to be already normalized

        weights = self.param(
            "weights",
            self.weight_init,
            (self.num_positions, self.vocab_size, self.layer_width),
        )
        # log_weights = weights

        # log_weights = log_softmax(weights, axis=1)
        # log_weights = log_weights - log_weights.max(axis=1, keepdims=True)
        # log_weights = jnp.minimum(log_weights, 0.0)

        # log_weights = weights - jnp.max(weights, axis=1, keepdims=True)
        log_weights = weights - jnp.max(weights, axis=1, keepdims=True)

        # FIXME: Is this all getting properly normalized?
        # logprob_weights = nn.relu(weights) + EPSILON

        # # element-wise product of weight vector and token vector for each column in the layer
        # rho = logprob_weights * t

        # # make it an inner product by taking a sum along the token dimension
        # rho = jnp.sum(rho, axis=1)  # after summing it is size = (num_positions, layer_width)

        # this replaces a prob domain element-wise product followed by sum on the axis=1
        rho = logsumexp(log_weights + t, axis=1)

        return rho  # rho is categorical
