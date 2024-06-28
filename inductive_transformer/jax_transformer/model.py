from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore

from inductive_transformer.jax_transformer.decoder_layer import DecoderLayer
from inductive_transformer.jax_transformer.encoder_layer import EncoderLayer
from inductive_transformer.jax_transformer.helper_functions import PROBABLE, IMPROBABLE


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
        # layer_t_categorical is the same copy in each layer_idx, so we can just grab the first one
        # same point on layer_width axis
        sentence_t_categorical = t_categorical[0, :, :, 0]
        assert sentence_t_categorical.shape == (self.num_positions, self.vocab_size)

        # Construct the embedding vector for the padding token.
        padding_embedding = jnp.full(self.vocab_size, IMPROBABLE)
        padding_embedding = padding_embedding.at[self.vocab_size - 1].set(PROBABLE)
        assert padding_embedding.shape == (self.vocab_size,)

        # We rely on broadcasting rules, which work right to left. The last axis for both arrays
        # has size vocab_size. Only t_categorical has another axis, so padding_embedding will
        # implicitly be treated as if it had a first axis of size 1.
        mask = sentence_t_categorical == padding_embedding
        # We're equal to the padding token only if we agree across all of the vocab size axis.
        mask = jnp.all(mask, axis=1)
        assert mask.shape == (self.num_positions,)

        # We want to know in which position the padding is
        for layer_idx, encoder in enumerate(self.encoders):
            # the words flow into the encoder in reverse order e.g. the sentence "big cat"
            # has word "big" flow to layer 1 and "cat" flow to layer 0
            # if "cat" was <padding> then we flow <padding> to layer 0
            layer_t_categorical = t_categorical[layer_idx]
            assert layer_t_categorical.shape == (self.num_positions, self.vocab_size, self.layer_width)
            z, x, y, activations = encoder(z, layer_t_categorical, mask[self.num_layers - layer_idx - 1])
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
        assert decoder_z.shape == (self.num_layers, 2, self.layer_width)
        decoder_t = jnp.stack(decoder_t, axis=0)
        assert decoder_t.shape == (self.num_layers, self.num_positions, self.vocab_size, self.layer_width)
        decoder_t = decoder_t.sum(axis=(0, -1))
        assert decoder_t.shape == (self.num_positions, self.vocab_size)

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
