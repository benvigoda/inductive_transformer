from torch import nn  # type: ignore

class EncoderLayer(nn.Module):

    def __init__(self,):
        super(EncoderLayer, self).__init__()

        self.encoder_universe = EncoderOpenClosedUniverse()
        self.encoder_bernoulli_categorical = EncoderBernoulliCategorical()
        self.encoder_token_pi = EncoderTokenPi()
        self.encoder_attention_pi = EncoderAttentionPi()
        self.encoder_categorical_bernoulli = EncoderCategoricalBernoulli()
        self.encoder_and = EncoderAnd()

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
