import torch  # type: ignore
from torch import nn  # type: ignore
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer


class Model(nn.Module):
    """
    dim=0 is always states of the variable e.g. 0,1 or cat, dog
    dim=1 is layer_width
    """
    def __init__(self, hyperparams):
        super(Model, self).__init__()
        self.hyperparams = hyperparams
        self.layer_width = hyperparams.layer_width
        self.vocab_size = hyperparams.vocab_size
        self.num_layers = hyperparams.num_layers

        self.encoder_layer_0 = EncoderLayer(hyperparams=hyperparams, active_layer=0)
        self.encoder_layer_1 = EncoderLayer(hyperparams=hyperparams, active_layer=1)

        self.decoder_layer_0 = DecoderLayer(hyperparams=hyperparams, active_layer=0)
        self.decoder_layer_1 = DecoderLayer(hyperparams=hyperparams, active_layer=1)

        # Tuple of variables output by the forward pass
        # This can then be easily accessed for printing
        self.encoder_output = tuple()
        self.decoder_output = tuple()

    # two layer model
    def forward(self, z_input, t):
        assert t.shape == (self.num_layers, self.vocab_size, self.layer_width)
        assert z_input.shape == (2, self.layer_width)
        z1_encode = self.encoder_layer_0(z_input, t[0])
        z2_encode = self.encoder_layer_1(z1_encode, t[1])

        z2_decode = z2_encode  # take the output of the encoder and feed it to the decoder

        t_decode_layer_1, z1_decode = self.decoder_layer_1(z2_decode)
        t_decode_layer_0, z0_decode = self.decoder_layer_0(z1_decode)

        self.encoder_output = (z1_encode, z2_encode)
        self.decoder_output = torch.cat([t_decode_layer_0, t_decode_layer_1, z0_decode], dim=0)

        return self.decoder_output

    '''
    one layer model
    def forward(self, z0):

        z1_encode = self.encoder_layer0(z0)

        z1_decode = z1_encode # take the output of the encoder and feed it to the decoder

        t_decode_layer_0, z0_decode = decoder_layer0(z1_decode)

        self.encoder_output = (z1_encode)
        self.decoder_output = (t_decode_layer_0, z0_decode)
    '''
