from torch import nn  # type: ignore


class EncoderAttentionPi(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderAttentionPi, self).__init__()
        self.hyperparams = hyperparams

    def forward(self, v):

        y = v #FIXME
        
        return y