from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore

from inductive_transformer.jax_transformer.decoder_universe import DecoderUniverse
from inductive_transformer.jax_transformer.decoder_bernoulli_categorical import (
    DecoderBernoulliCategorical,
)
from inductive_transformer.jax_transformer.decoder_categorical_bernoulli import (
    DecoderCategoricalBernoulli,
)
from inductive_transformer.jax_transformer.decoder_token_pi import DecoderTokenPi
from inductive_transformer.jax_transformer.decoder_position_pi import DecoderPositionPi
from inductive_transformer.jax_transformer.decoder_attention_pi import (
    DecoderAttentionPi,
)
from inductive_transformer.jax_transformer.decoder_and import DecoderAnd


class DecoderLayer(nn.Module):
    layer_width: int
    num_positions: int
    vocab_size: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)
    use_encoder_message: bool = True

    def setup(self):
        self.decoder_universe = DecoderUniverse(layer_width=self.layer_width)
        self.decoder_bernoulli_categorical = DecoderBernoulliCategorical(
            layer_width=self.layer_width
        )
        self.decoder_token_pi = DecoderTokenPi(
            num_positions=self.num_positions,
            vocab_size=self.vocab_size,
            layer_width=self.layer_width,
            weight_init=self.weight_init,
        )
        self.decoder_attention_pi = DecoderAttentionPi(
            layer_width=self.layer_width, weight_init=self.weight_init
        )
        self.decoder_position_pi = DecoderPositionPi(
            num_positions=self.num_positions,
            layer_width=self.layer_width,
        )
        self.decoder_categorical_bernoulli = DecoderCategoricalBernoulli(
            layer_width=self.layer_width
        )
        self.decoder_and = DecoderAnd(
            layer_width=self.layer_width, use_encoder_message=self.use_encoder_message
        )

    def __call__(self, z_prime, x_encoder, y_encoder):
        # dim=0 indexes the state of the variable e.g. cat or dog, 0 or 1, etc.
        # dim=1 indexes the layer width

        assert z_prime.shape == (2, self.layer_width)
        assert x_encoder.shape == (2, self.layer_width)
        assert y_encoder.shape == (2, self.layer_width)

        # Decoder $\land$
        x_bernoulli, y_bernoulli = self.decoder_and(z_prime, x_encoder, y_encoder)

        # Decoder Bernoulli-Categorical
        y_categorical = self.decoder_bernoulli_categorical(y_bernoulli)
        x_categorical = self.decoder_bernoulli_categorical(x_bernoulli)

        # Decoder Attention $\pi$
        v = self.decoder_attention_pi(y_categorical)

        rho_categorical = self.decoder_position_pi(x_categorical)

        # Decoder Word $\pi$
        t_categorical = self.decoder_token_pi(rho_categorical)

        # Decoder Categorical-Bernoulli
        # u is bernoulli, v is categorical
        u = self.decoder_categorical_bernoulli(v)

        # Decoder Open Closed Universe
        z = self.decoder_universe(u)

        activations = {
            "x_bernoulli": x_bernoulli,
            "y_bernoulli": y_bernoulli,
            "x_categorical": x_categorical,
            "y_categorical": y_categorical,
            "v": v,
            "rho_categorical": rho_categorical,
            "t_categorical": t_categorical,
            "u": u,
            "z": z,
        }
        return z, t_categorical, activations
