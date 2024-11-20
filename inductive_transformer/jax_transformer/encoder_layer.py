from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore
import jax

from inductive_transformer.jax_transformer.encoder_universe import EncoderUniverse
from inductive_transformer.jax_transformer.encoder_bernoulli_categorical import (
    EncoderBernoulliCategorical,
)
from inductive_transformer.jax_transformer.encoder_token_pi import EncoderTokenPi
from inductive_transformer.jax_transformer.encoder_position_pi import EncoderPositionPi
from inductive_transformer.jax_transformer.encoder_attention_pi import (
    EncoderAttentionPi,
)
from inductive_transformer.jax_transformer.encoder_categorical_bernoulli import (
    EncoderCategoricalBernoulli,
)
from inductive_transformer.jax_transformer.encoder_and import EncoderAnd


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
        u_nan = jnp.isnan(u).any()
        try:
            if u_nan.val[0]:
                print("nan in u at encoder_layer")
        except:
            if u_nan:
                print("nan in u at encoder_layer")

        # Encoder Bernoulli-Categorical
        # u is bernoulli, v is categorical
        v = self.encoder_bernoulli_categorical(u)
        v_nan = jnp.isnan(v).any()
        try:
            if v_nan.val[0]:
                print("nan in v at encoder_layer")
        except:
            if v_nan:
                print("nan in v at encoder_layer")

        # u and v appear to be fine

        # Encoder Attention $\pi$
        y_categorical = self.encoder_attention_pi(v)
        assert y_categorical.shape == (1, self.layer_width)
        y_categorical_nan = jnp.isnan(y_categorical).any()
        try:
            if y_categorical_nan.val[0]:
                print("nan in y_categorical at encoder_layer")
        except:
            if y_categorical_nan:
                print("nan in y_categorical at encoder_layer")

        y_bernoulli = self.encoder_categorical_bernoulli(y_categorical)
        assert y_bernoulli.shape == (2, self.layer_width)
        y_bernoulli_nan = jnp.isnan(y_bernoulli).any()
        try:
            if y_bernoulli_nan.val[0]:
                print("nan in y_bernoulli at encoder_layer")
        except:
            if y_bernoulli_nan:
                print("nan in y_bernoulli at encoder_layer")

        # import pdb; pdb.set_trace()

        # Hook 3 pi_t's to their parent pi_rho, everywhere this occurs.
        # The encoder open-closed universe without backwards info from the decoder simply clones the input data for a given token and position
        # and sends it to the corresponding pi_t in both the the left and right columns.
        # We simply make the input data have the same values for every value of lw (which indexes layer_width).
        rho_categorical = self.encoder_token_pi(t_categorical)
        # This is one place where the nan's are coming from
        # But there are nan's earlier in the code as well
        if jnp.isnan(rho_categorical).any().val[0]:
            print("nan in rho_categorical at encoder_layer")
            import pdb; pdb.set_trace()
        # Encoder Position $\pi$
        # without position, we had pi_t outputting x_categorical, with
        # t_categorical.shape() = (vocab_size, layer_width)
        # now this is the job of pi_rho which is replacing pi_t, so we have this:
        assert rho_categorical.shape == (self.num_positions, self.layer_width)

        # Encoder Word $\pi$
        x_categorical = self.encoder_position_pi(rho_categorical)
        assert x_categorical.shape == (1, self.layer_width)

        # Encoder Categorical-Bernoulli
        x_bernoulli = self.encoder_categorical_bernoulli(x_categorical)
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
        # When padding we do not know which word should be activated, so we set all to 0.5
        masked_z = jnp.ones(shape=(2, self.layer_width)) * 0.5
        assert masked_z.shape == (2, self.layer_width)

        z_prime = jnp.where(masked, masked_z, z_prime)

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
        if jnp.isnan(z_prime).any().val[0]:
            print("nan in z_prime at encoder_layer")
            import pdb; pdb.set_trace()
        if jnp.isnan(x_bernoulli).any().val[0]:
            print("nan in x_bernoulli at encoder_layer")
            import pdb; pdb.set_trace()
        y_nan = jnp.isnan(y_bernoulli).any()
        try:
            if y_nan.val[0]:
                print("nan in y_bernoulli at encoder_layer")
                import pdb; pdb.set_trace()
        except:
            if y_nan:
                print("nan in y_bernoulli at encoder_layer")
                import pdb; pdb.set_trace()

        return z_prime, x_bernoulli, y_bernoulli, activations
