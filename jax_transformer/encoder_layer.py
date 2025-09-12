# Copyright 2025 Ben Vigoda, Thomas Rochais, and Erik Strand
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore

from jax_transformer.encoder_universe import EncoderUniverse
from jax_transformer.encoder_bernoulli_categorical import (
    EncoderBernoulliCategorical,
)
from jax_transformer.encoder_token_pi import EncoderTokenPi
from jax_transformer.encoder_position_pi import EncoderPositionPi
from jax_transformer.encoder_attention_pi import (
    EncoderAttentionPi,
)
from jax_transformer.encoder_categorical_bernoulli import (
    EncoderCategoricalBernoulli,
)
from jax_transformer.encoder_and import EncoderAnd


# In terms of left side pi_t's
# We need 3 left pi_t's going to the left pi_rho, one for each of the positions
# PLUS 3 left pi_t's going to the right pi_rho, one for each of the positions


class EncoderLayer(nn.Module):
    layer_width: int
    num_positions: int
    vocab_size: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    def setup(self):
        self.encoder_universe = EncoderUniverse(layer_width=self.layer_width)
        self.encoder_bernoulli_categorical = EncoderBernoulliCategorical()
        self.encoder_token_pi = EncoderTokenPi(
            num_positions=self.num_positions,
            vocab_size=self.vocab_size,
            layer_width=self.layer_width,
            weight_init=self.weight_init,
        )
        self.encoder_position_pi = EncoderPositionPi(
            num_positions=self.num_positions,
            layer_width=self.layer_width,
            weight_init=self.weight_init,
        )
        self.encoder_attention_pi = EncoderAttentionPi(
            vocab_size=self.vocab_size,
            layer_width=self.layer_width,
            weight_init=self.weight_init,
        )
        self.encoder_categorical_bernoulli = EncoderCategoricalBernoulli(
            layer_width=self.layer_width
        )
        self.encoder_and = EncoderAnd()

    def __call__(self, z, t_categorical, masked):
        assert z.shape == (2, self.layer_width)
        assert t_categorical.shape == (
            self.num_positions,
            self.vocab_size,
            self.layer_width,
        )

        # axis=0 indexes the state of the variable e.g. cat or dog, 0 or 1, etc.
        # axis=1 indexes the layer width

        # Encoder Open Closed Universe
        u = self.encoder_universe(z)
        assert u.shape == (2, self.layer_width, self.layer_width)

        # Encoder Bernoulli-Categorical
        # u is bernoulli, v is categorical
        v = self.encoder_bernoulli_categorical(u)

        # Encoder Attention $\pi$
        y_categorical = self.encoder_attention_pi(v)
        assert y_categorical.shape == (1, self.layer_width)

        y_bernoulli = self.encoder_categorical_bernoulli(y_categorical, normalize=True)
        assert y_bernoulli.shape == (2, self.layer_width)

        # Hook 3 pi_t's to their parent pi_rho, everywhere this occurs.
        # The encoder open-closed universe without backwards info from the decoder simply clones the input data for a given token and position
        # and sends it to the corresponding pi_t in both the the left and right columns.
        # We simply make the input data have the same values for every value of lw (which indexes layer_width).
        rho_categorical = self.encoder_token_pi(t_categorical)
        # Encoder Position $\pi$
        # without position, we had pi_t outputting x_categorical, with
        # t_categorical.shape() = (vocab_size, layer_width)
        # now this is the job of pi_rho which is replacing pi_t, so we have this:
        assert rho_categorical.shape == (self.num_positions, self.layer_width)

        # Encoder Word $\pi$
        x_categorical = self.encoder_position_pi(rho_categorical)
        assert x_categorical.shape == (1, self.layer_width)

        # Encoder Categorical-Bernoulli
        x_bernoulli = self.encoder_categorical_bernoulli(x_categorical, normalize=False)
        assert x_bernoulli.shape == (2, self.layer_width)

        # Encoder $\land$
        z_prime = self.encoder_and(x_bernoulli, y_bernoulli)
        assert z_prime.shape == (2, self.layer_width)

        """
        OLD WAY of thinking
            # if we are in an encoder layer where text_parsing tells us there is no input text from the prompt
            # and the input token is <padding> then set the output of the encoder layer to all True

            # If layer_t_categorical[num_layers-i-1,:,0] == padding_embedding,
            #   set z[0, :] = 0
            #   set z[1, :] = 1
            # Talking about the output z of the encoder, so really z_prime
            masked_z = jnp.stack(
                [
                    jnp.zeros(shape=(self.layer_width,)),
                    jnp.ones(shape=(self.layer_width,)),
                ],
                axis=0,
            )
        """

        """
        August 2025 update:
        Whenever there is padding, we needed the x_encoder being sent to the decoder_equals to enable it to pass the z_decoder through to y_decoder uninterrupted.
        That means x_encoder Bernoulli needed to be 50-50.  So now we do that.  Any padding in the prompt results in the x_encoder signal being a 50-50.
        This works for us where we have one token per layer.
        In the future, when our attention_pi or token_pi systems become a binary tree (two-hot tree) instead of a one-hot tree, we will have two words per layer and we will be forced to revisit how we do this.
        In particular, we won't be able to assume a 1:1 correspondence between layers and positions in the text.
        We shouldn't really be able to assume that now, because in theory any of our layers could learn to output to any position in the text.
        But forward masking in a vanilla transformer masks that off and enforces this kind of assumption, and so do we.
        Right now we essentially have a 1x num_layers vector that maps learned positions to layers, or if you prefer we have a num_layers x num_layers permutation matrix doing this mapping with a single 1 per row and column.
        In the future this will become a full attention matrix representing pair-wise joint attention.  And it will need forward masking.
        """

        # When padding we do not know which word should be activated, so we set all to log(0.5)
        masked_z = jnp.ones(shape=(2, self.layer_width)) * jnp.log(0.5)
        assert masked_z.shape == (2, self.layer_width)

        z_prime = jnp.where(masked, masked_z, z_prime)

        # Do the same thing for x_categorical and x_bernoulli
        masked_x_categorical = jnp.ones(shape=(1, self.layer_width)) * jnp.log(1/self.layer_width)
        masked_x_bernoulli = jnp.ones(shape=(2, self.layer_width)) * jnp.log(0.5)
        x_categorical = jnp.where(masked, masked_x_categorical, x_categorical)
        x_bernoulli = jnp.where(masked, masked_x_bernoulli, x_bernoulli)

        activations = {
            "z": z,
            "u": u,
            "v": v,
            "y_categorical": y_categorical,
            "y_bernoulli": y_bernoulli,
            "rho_categorical": rho_categorical,
            "x_categorical": x_categorical,
            "x_bernoulli": x_bernoulli,
            "z_prime": z_prime,
        }

        return z_prime, x_bernoulli, y_bernoulli, activations
