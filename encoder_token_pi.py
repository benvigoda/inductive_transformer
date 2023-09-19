from torch import nn  # type: ignore


class EncoderTokenPi(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderTokenPi, self).__init__()
        self.hyperparams = hyperparams

    def forward(self, t):

        t = w*x #FIXME
        
        return x