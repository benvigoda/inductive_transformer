from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore
from jax_transformer.helper_functions import custom_normalize


@dataclass
class EncoderCategoricalBernoulli:
    layer_width: int

    def __call__(self, categorical):
        # categorical is size = (1, layer_width)
        assert categorical.shape == (1, self.layer_width)

        categorical = custom_normalize(categorical, axis=1)

        # bernoulli is size (2, layer_width)
        bernoulli_1 = categorical
        bernoulli_0 = (
            jnp.log1p(jnp.exp(-categorical))
        )

        bernoulli = jnp.concatenate([bernoulli_0, bernoulli_1])

        # by construction these are each normalized
        # bernoulli = custom_normalize(bernoulli, axis=0)
        assert bernoulli.shape == (2, self.layer_width)

        return bernoulli


        # The probability of a bernoulli variable being true is the same as the probability of the
        # corresponding categorical state.

        # the assumption is that the categorical coming in is properly normalized
        # but to we should verify that each output from each attention pi's is between 0 and 1
        # Check that the categorical is between 0 and 1
        # def assert_all_in_range(categorical):
        #     def true_fn(_):
        #         return True

        #     def false_fn(_):
        #         raise ValueError("Categorical values must be in the range [0, 1]")

        #     jax.lax.cond(jnp.all(categorical >= 0) & jnp.all(categorical <= 1),
        #                 true_fn,
        #                 false_fn,
        #                 None)

        # # Perform the assertion
        # assert_all_in_range(categorical)

        # print("All values are in the range [0, 1].")
