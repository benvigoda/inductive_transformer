from dataclasses import dataclass

from jax_transformer.helper_functions import custom_normalize, EPSILON, bound_activations


@dataclass
class EncoderBernoulliCategorical:
    def __call__(self, u):
        # v = u[1] / (u[0] + EPSILON)
        # if u is properly normalized then we should not need to divide by zero
        v = u[1]

        v = bound_activations(v)
        return v
