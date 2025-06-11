from torch import nn  # type: ignore
from inductive_transformer.torch_transformer.encoder_universe import EncoderUniverse
from inductive_transformer.torch_transformer.encoder_bernoulli_categorical import (
    EncoderBernoulliCategorical,
)
from inductive_transformer.torch_transformer.encoder_token_pi import EncoderTokenPi
from inductive_transformer.torch_transformer.encoder_position_pi import (
    EncoderPositionPi,
)
from inductive_transformer.torch_transformer.encoder_attention_pi import (
    EncoderAttentionPi,
)
from inductive_transformer.torch_transformer.encoder_categorical_bernoulli import (
    EncoderCategoricalBernoulli,
)
from inductive_transformer.torch_transformer.encoder_and import EncoderAnd


# In terms of left side pi_t's
# We need 3 left pi_t's going to the left pi_rho, one for each of the positions
# PLUS 3 left pi_t's going to the right pi_rho, one for each of the positions


class EncoderLayer(nn.Module):
    def __init__(self, hyperparams, active_layer: int):
        super(EncoderLayer, self).__init__()
        self.active_layer = active_layer
        self.hyperparams = hyperparams
        self.encoder_universe = EncoderUniverse(
            hyperparams=hyperparams, active_layer=active_layer
        )
        self.encoder_bernoulli_categorical = EncoderBernoulliCategorical(
            hyperparams=hyperparams, active_layer=active_layer
        )
        self.encoder_token_pi = EncoderTokenPi(
            hyperparams=hyperparams, active_layer=active_layer
        )
        self.encoder_position_pi = EncoderPositionPi(
            hyperparams=hyperparams, active_layer=active_layer
        )
        self.encoder_attention_pi = EncoderAttentionPi(
            hyperparams=hyperparams, active_layer=active_layer
        )
        self.encoder_categorical_bernoulli = EncoderCategoricalBernoulli(
            hyperparams=hyperparams, active_layer=active_layer
        )
        self.encoder_and = EncoderAnd(
            hyperparams=hyperparams, active_layer=active_layer
        )

        self.z_prime = None

    def forward(self, z, t_categorical):
        # dim=0 indexes the state of the variable e.g. cat or dog, 0 or 1, etc.
        # dim=1 indexes the layer width

        # Encoder Open Closed Universe
        u = self.encoder_universe(z)

        # Encoder Bernoulli-Categorical
        # u is bernoulli, v is categorical
        v = self.encoder_bernoulli_categorical(u)

        # Encoder Attention $\pi$
        y_categorical = self.encoder_attention_pi(v)
        assert y_categorical.shape == (1, self.hyperparams.layer_width)

        # Encoder Word $\pi$
        # without position, we had
        # t_categorical.shape() = (vocab_size, layer_width)
        # now with position we have:
        assert t_categorical.shape == (
            self.hyperparams.num_positions,
            self.hyperparams.vocab_size,
            self.hyperparams.layer_width,
        )

        # Hook 3 pi_t's to their parent pi_rho, everywhere this occurs.
        # The encoder open-closed universe without backwards info from the decoder simply clones the input data for a given token and position
        # and sends it to the corresponding pi_t in both the the left and right columns.
        # We simply make the input data have the same values for every value of lw (which indexes layer_width).
        rho_categorical = self.encoder_token_pi(t_categorical)
        # Encoder Position $\pi$
        # without position, we had pi_t outputting x_categorical, with
        # t_categorical.shape() = (vocab_size, layer_width)
        # now this is the job of pi_rho which is replacing pi_t, so we have this:
        assert rho_categorical.shape == (
            self.hyperparams.num_positions,
            self.hyperparams.layer_width,
        )

        # Encoder Word $\pi$
        x_categorical = self.encoder_position_pi(rho_categorical)
        assert x_categorical.shape == (1, self.hyperparams.layer_width)

        # Encoder Categorical-Bernoulli
        y_bernoulli = self.encoder_categorical_bernoulli(y_categorical)
        x_bernoulli = self.encoder_categorical_bernoulli(x_categorical)

        # Encoder $\land$
        z_prime = self.encoder_and(x_bernoulli, y_bernoulli)
        self.z_prime = z_prime

        return z_prime
