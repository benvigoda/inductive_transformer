from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore
from jax_transformer.helper_functions import custom_normalize
import jax

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

        # The probability of a bernoulli variable being true is the same as the probability of the
        # corresponding categorical state.

        # the assumption is that the categorical coming in is properly normalized
        # but to we should verify that each output from each attention pi's is less than 0

        # def assert_all_in_range(categorical):
        #     # Print the actual values and the comparison result
        #     jax.debug.print("Categorical values: {}", categorical)
        #     comparison = categorical < 0
        #     jax.debug.print("Less than zero check: {}", comparison)
        #     jax.debug.print("All less than zero: {}", jnp.all(comparison))

        #     # Store the condition result in a variable
        #     all_less_than_zero = jnp.all(categorical < 0)

        #     # Use a different approach with custom message
        #     jax.debug.print("Condition result: {}", all_less_than_zero)

        #     # No conditional logic, just return True
        #     return True

        # # # Perform the assertion
        # assert_all_in_range(categorical)

        return bernoulli
