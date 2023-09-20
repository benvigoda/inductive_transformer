
from torch import nn  # type: ignore
from decoder_open_closed_universe import DecoderOpenClosedUniverse
from decoder_bernoulli_categorical import DecoderBernoulliCategorical
from decoder_categorical_bernoulli import DecoderCategoricalBernoulli
from decoder_token_pi import DecoderTokenPi
from decoder_attention_pi import DecoderAttentionPi
from decoder_and import DecoderAnd

class DecoderLayer(nn.Module):

    def __init__(self, hyperparams):
        super(DecoderLayer, self).__init__()

        self.decoder_universe = DecoderOpenClosedUniverse(hyperparams)
        self.decoder_bernoulli_categorical = DecoderBernoulliCategorical(hyperparams)
        self.decoder_token_pi = DecoderTokenPi(hyperparams)
        self.decoder_attention_pi = DecoderAttentionPi(hyperparams)
        self.decoder_categorical_bernoulli = DecoderCategoricalBernoulli(hyperparams)
        self.decoder_and = DecoderAnd(hyperparams)

    def forward(self, z_prime):
        # dim=0 indexes the state of the variable e.g. cat or dog, 0 or 1, etc.
        # dim=1 indexes the layer width

        # Decoder $\land$
        x_bernoulli, y_bernoulli = self.decoder_and(z_prime)

        # Decoder Bernoulli-Categorical
        y_categorical = self.decoder_bernoulli_categorical(y_bernoulli)
        x_categorical = self.decoder_bernoulli_categorical(x_bernoulli)

        # Decoder Attention $\pi$
        v = self.decoder_attention_pi(y_categorical)

        # Decoder Word $\pi$
        t = self.decoder_token_pi(x_categorical)

        # Decoder Categorical-Bernoulli
        # u is bernoulli, v is categorical
        u = self.decoder_categorical_bernoulli(v)

        # Decoder Open Closed Universe
        z = self.decoder_universe(u)

        return z
