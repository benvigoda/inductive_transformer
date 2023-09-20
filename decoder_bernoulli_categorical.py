from torch import nn  # type: ignore


class DecoderBernoulliCategorical(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderBernoulliCategorical, self).__init__()
        self.hyperparams = hyperparams

    def forward(self, bernoulli):
        # categorical is size = (1, layer_width)
        # bernoulli is size (2, layer_width)
        categorical[0, 0] = bernoulli[1][0]
        categorical[0, 0] = bernoulli[0][1]
        

        categorical[0, 1] = bernoulli[0][0]
        categorical[0, 1] = bernoulli[1][1]

        categorical = torch.normalize(categorical, p=1, dim=1)

        return categorical
        
        