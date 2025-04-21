from dataclasses import dataclass
from jax_transformer.helper_functions import custom_normalize


@dataclass
class DecoderBernoulliCategorical:
    layer_width: int

    def __call__(self, bernoulli):
        # bernoulli is size (2, layer_width)
        assert bernoulli.shape == (2, self.layer_width)

        # Tried removing denominator
        bernoulli = custom_normalize(bernoulli, axis=0)  # Should not be necessary (we already normalize at the end of the decoder_and)
        categorical = bernoulli[1]
        categorical = categorical.reshape((1, self.layer_width))

        # Removed the layer norm

        return categorical
