from torch import nn  # type: ignore
from encoder_open_closed_universe import EncoderOpenClosedUniverse
from encoder_bernoulli_categorical import EncoderBernoulliCategorical
from encoder_token_pi import EncoderTokenPi
from encoder_attention_pi import EncoderAttentionPi
from encoder_categorical_bernoulli import EncoderCategoricalBernoulli
from encoder_and import EncoderAnd


class EncoderLayer(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderLayer, self).__init__()

        self.encoder_universe = EncoderOpenClosedUniverse(hyperparams)
        self.encoder_bernoulli_categorical = EncoderBernoulliCategorical(hyperparams)
        self.encoder_token_pi = EncoderTokenPi(hyperparams)
        self.encoder_attention_pi = EncoderAttentionPi(hyperparams)
        self.encoder_categorical_bernoulli = EncoderCategoricalBernoulli(hyperparams)
        self.encoder_and = EncoderAnd(hyperparams)

    def forward(self, z, t):
        # dim=0 indexes the state of the variable e.g. cat or dog, 0 or 1, etc.
        # dim=1 indexes the layer width

        # Encoder Open Closed Universe
        u = self.encoder_universe(z)

        # Encoder Bernoulli-Categorical
        # u is bernoulli, v is categorical
        v = self.encoder_bernoulli_categorical(u)

        # Encoder Attention $\pi$
        y_categorical = self.encoder_attention_pi(v)

        # Encoder Word $\pi$
        x_categorical = self.encoder_token_pi(t)

        # Encoder Categorical-Bernoulli
        y_bernoulli = self.encoder_categorical_bernoulli(y_categorical)
        x_bernoulli = self.encoder_categorical_bernoulli(x_categorical)

        # Encoder $\land$
        z_prime = self.encoder_and(x_bernoulli, y_bernoulli)

        return z_prime
