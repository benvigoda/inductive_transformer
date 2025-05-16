from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore
import jax.nn as nn
from jax_transformer.helper_functions import bound_activations

@dataclass
class DecoderCategoricalBernoulli:
    layer_width: int

    def __call__(self, v):
        assert v.shape == (self.layer_width, self.layer_width)

        # The probability of a bernoulli variable being True is the same as the probability of the
        # corresponding categorical state.
        u_1 = v

        # The probability of a bernoulli variable being False is 1 - the probability of it being True.
        # u_0 = 1.0 - u_1
        u_0 = jnp.log1p(-u_1)
        # u_0 = nn.log_sigmoid(-u_1)

        u = jnp.stack([u_0, u_1], axis=0)

        assert u.shape == (2, self.layer_width, self.layer_width)

        u = bound_activations(u)
        return u