from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from jax_transformer.helper_functions import custom_normalize


@dataclass
class EncoderCategoricalBernoulli:
    layer_width: int

    def __call__(self, categorical):

        # categorical is size = (1, layer_width)
        assert categorical.shape == (1, self.layer_width)
        # bernoulli is size (2, layer_width)

        # The probability of a bernoulli variable being true is the same as the probability of the
        # corresponding categorical state.

        bernoulli_1 = categorical

        # The probability of a bernoulli variable being false is the sum of the probabilities of all
        # the other categorical states.
        # Note: if categorical[i][j] is much larger than categorical[i][k] for k != j, then this
        # method of performing the calculation introduces a lot of rounding error.

        bernoulli_0 = categorical.sum(axis=-1, keepdims=True) - categorical
        bernoulli = jnp.concatenate([bernoulli_0, bernoulli_1])
        bernoulli = custom_normalize(bernoulli, axis=0)
        assert bernoulli.shape == (2, self.layer_width)

        return bernoulli
