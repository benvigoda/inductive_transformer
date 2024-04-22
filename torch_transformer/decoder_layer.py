from torch import nn  # type: ignore
from torch_transformer.decoder_universe import DecoderUniverse
from torch_transformer.decoder_bernoulli_categorical import DecoderBernoulliCategorical
from torch_transformer.decoder_categorical_bernoulli import DecoderCategoricalBernoulli
from torch_transformer.decoder_token_pi import DecoderTokenPi
from torch_transformer.decoder_position_pi import DecoderPositionPi
from torch_transformer.decoder_attention_pi import DecoderAttentionPi
from torch_transformer.decoder_and import DecoderAnd


class DecoderLayer(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderLayer, self).__init__()
        self.active_layer = active_layer
        self.decoder_universe = DecoderUniverse(hyperparams=hyperparams, active_layer=active_layer)
        self.decoder_bernoulli_categorical = DecoderBernoulliCategorical(hyperparams=hyperparams, active_layer=active_layer)
        self.decoder_token_pi = DecoderTokenPi(hyperparams=hyperparams, active_layer=active_layer)
        self.decoder_attention_pi = DecoderAttentionPi(hyperparams=hyperparams, active_layer=active_layer)
        self.decoder_position_pi = DecoderPositionPi(hyperparams=hyperparams, active_layer=active_layer)
        self.decoder_categorical_bernoulli = DecoderCategoricalBernoulli(hyperparams=hyperparams, active_layer=active_layer)
        self.decoder_and = DecoderAnd(hyperparams=hyperparams, active_layer=active_layer)

    def forward(self, z_prime, x_encoder, y_encoder):
        # dim=0 indexes the state of the variable e.g. cat or dog, 0 or 1, etc.
        # dim=1 indexes the layer width

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

        return t_categorical, z
