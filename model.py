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
        self.num_positions = hyperparams.num_positions

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
        assert t.shape == (self.num_layers, self.num_positions, self.vocab_size, self.layer_width)
        assert z_input.shape == (2, self.layer_width)
        if t[1].numel() == 0:
            z1_encode = z_input
            x_encoder_0 = None
            y_encoder_0 = None
        else:
            z1_encode = self.encoder_layer_0(z_input, t[1])
            # passing x and y from the encoder to the decoder
            x_encoder_0 = self.encoder_layer_0.encoder_and.x
            y_encoder_0 = self.encoder_layer_0.encoder_and.y

        z2_encode = self.encoder_layer_1(z1_encode, t[0])

        # passing x and y from the encoder to the decoder
        x_encoder_1 = self.encoder_layer_1.encoder_and.x
        y_encoder_1 = self.encoder_layer_1.encoder_and.y

        z2_decode = z2_encode  # take the output of the encoder and feed it to the decoder

        t_decode_layer_1, z1_decode = self.decoder_layer_1(z2_decode, x_encoder_1, y_encoder_1)
        t_decode_layer_0, z0_decode = self.decoder_layer_0(z1_decode, x_encoder_0, y_encoder_0)

        # TODO: The new shapes should include num_positions once the decoder is position sensitive
        assert t_decode_layer_0.shape == (self.vocab_size, self.layer_width)
        assert t_decode_layer_1.shape == (self.vocab_size, self.layer_width)

        self.encoder_output = (z1_encode, z2_encode)
        self.decoder_output = torch.stack([t_decode_layer_0, t_decode_layer_1], dim=0)
        assert self.decoder_output.shape == (self.num_layers, self.vocab_size, self.layer_width)
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
