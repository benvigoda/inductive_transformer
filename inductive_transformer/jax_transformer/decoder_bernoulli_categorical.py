from dataclasses import dataclass

from inductive_transformer.jax_transformer.helper_functions import custom_normalize


@dataclass
class DecoderBernoulliCategorical:
    layer_width: int

    def __call__(self, bernoulli):
        # bernoulli is size (2, layer_width)
        assert bernoulli.shape == (2, self.layer_width)

        categorical = bernoulli[1] / (bernoulli[0] + 1e-9)
        categorical = categorical.reshape((1, self.layer_width))
        categorical = custom_normalize(categorical, axis=1)  # FIXME : Do we need this?

        return categorical
