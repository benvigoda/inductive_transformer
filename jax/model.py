from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore

from decoder_layer import DecoderLayer
from encoder_layer import EncoderLayer


class InductiveTransformer(nn.Module):
    layer_width: int
    num_positions: int
    vocab_size: int
    num_layers: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)
    use_encoder_message: bool = True

    def setup(self):
        self.encoders = [EncoderLayer(
            layer_width=self.layer_width,
            num_positions=self.num_positions,
            vocab_size=self.vocab_size,
            weight_init=self.weight_init,
        ) for _ in range(self.num_layers)]

        self.decoders = [DecoderLayer(
            layer_width=self.layer_width,
            num_positions=self.num_positions,
            vocab_size=self.vocab_size,
            weight_init=self.weight_init,
            use_encoder_message=self.use_encoder_message
        ) for _ in range(self.num_layers)]

    def __call__(self, z, t_categorical):
        """
        This is the forward pass of the model, defined without batches.
        """

        assert z.shape == (2, self.layer_width)
        assert t_categorical.shape == (self.num_layers, self.num_positions, self.vocab_size, self.layer_width)

        encoder_z = []
        encoder_x = []  # bernoulli
        encoder_y = []  # bernoulli
        encoder_activations = []
        for idx, encoder in enumerate(self.encoders):
            z, x, y, activations = encoder(z, t_categorical[idx])
            assert z.shape == (2, self.layer_width)
            assert x.shape == (2, self.layer_width)
            assert y.shape == (2, self.layer_width)
            encoder_z.append(z)
            encoder_x.append(x)
            encoder_y.append(y)
            encoder_activations.append(activations)

        decoder_z = [None] * (self.num_layers)
        decoder_t = [None] * (self.num_layers)
        decoder_activations = [None] * (self.num_layers)
        for idx in range(self.num_layers - 1, -1, -1):
            decoder = self.decoders[idx]
            z, t, activations = decoder(z, encoder_x[idx], encoder_y[idx])
            assert z.shape == (2, self.layer_width)
            assert t.shape == (self.num_positions, self.vocab_size, self.layer_width)
            decoder_z[idx] = z
            decoder_t[idx] = t
            decoder_activations[idx] = activations

        decoder_z = jnp.stack(decoder_z, axis=0)
        decoder_t = jnp.stack(decoder_t, axis=0)
        assert decoder_z.shape == (self.num_layers, 2, self.layer_width)
        assert decoder_t.shape == (self.num_layers, self.num_positions, self.vocab_size, self.layer_width)
        return decoder_z, decoder_t, encoder_activations, decoder_activations


# JAX vmap takes a function and maps it over an additional axis.
# Flax has a lifted version of vmap that works with modules.
# We use the latter here to add a batch axis.
BatchedInductiveTransformer = nn.vmap(
    InductiveTransformer,
    in_axes=(None, 0),
    out_axes=(0, 0, 0, 0),
    variable_axes={'params': None},
    split_rngs={'params': False},
)
