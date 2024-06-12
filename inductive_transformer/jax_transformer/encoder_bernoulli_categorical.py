from dataclasses import dataclass
from inductive_transformer.jax_transformer.helper_functions import custom_normalize, EPSILON


@dataclass
class EncoderBernoulliCategorical:

    def __call__(self, u):

        # v = u[1] / (u[0] + EPSILON)
        # if u is properly normalized then we should not need to divide by zero
        v = u[1]


        # in the future we may want to normalize is the inputs to a specific attention pi, 
        #    remember from the encoder universe factor:
        # v[0][0] + v[1][0] = 1
        # v[0][1] + v[1][1] = 1
        # v = custom_normalize(v, axis=0)???
        
        return v
