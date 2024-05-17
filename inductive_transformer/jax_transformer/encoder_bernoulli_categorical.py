from dataclasses import dataclass
from inductive_transformer.jax_transformer.helper_functions import custom_normalize, EPSILON


@dataclass
class EncoderBernoulliCategorical:

    def __call__(self, u):

        # there's four coins coming in
        # to convert coins to categorical, it's always head divided by tails
        # and then normalize the categoricals
        # v[below_lw][above_lw] = u[heads][below_lw][above_lw] / u[tails][below_lw][above_lw]

        v = u[1] / (u[0] + EPSILON)

        # we want to normalize is the inputs to a specific pi_a, remember from the encoder universe factor:
        # v[0][0] + v[1][0] = 1
        # v[0][1] + v[1][1] = 1

        v = custom_normalize(v, axis=0)
        return v
