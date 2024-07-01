from dataclasses import dataclass

from inductive_transformer.jax_transformer.helper_functions import custom_normalize, EPSILON


@dataclass
class DecoderBernoulliCategorical:
    layer_width: int

    def __call__(self, bernoulli):
        # bernoulli is size (2, layer_width)
        assert bernoulli.shape == (2, self.layer_width)

        #tried removing denominator
        bernoulli = custom_normalize(bernoulli, axis=0)  
        categorical = bernoulli[1] #/ (bernoulli[0] + EPSILON) 
        categorical = categorical.reshape((1, self.layer_width))

        # Removed the layer norm

        return categorical
