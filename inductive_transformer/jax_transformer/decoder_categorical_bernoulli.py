from dataclasses import dataclass
from inductive_transformer.jax_transformer.helper_functions import custom_normalize
import jax.numpy as jnp  # type: ignore


@dataclass
class DecoderCategoricalBernoulli:
    layer_width: int

    def __call__(self, v):
        assert v.shape == (self.layer_width, self.layer_width)

        # The probability of a bernoulli variable being True is the same as the probability of the
        # corresponding categorical state.
        u_1 = v

        # The probability of a bernoulli variable being False is 1 - the probability of it being True.
        u_0 = 1.0 - u_1

        u = jnp.stack([u_0, u_1], axis=0)

        assert u.shape == (2, self.layer_width, self.layer_width)
        return u
