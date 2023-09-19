from torch import nn  # type: ignore


class Model(nn.Module):

    def __init__(self, hyperparams):

        self.layer_width = hyperparams.layer_width
        self.num_layers = hyperparams.num_layers

        self.encoder_layer_0 = EncoderLayer(hyperparams)
        self.encoder_layer_1 = EncoderLayer(hyperparams)

        self.decoder_layer_0 = DecoderLayer(hyperparams)
        self.decoder_layer_1 = DecoderLayer(hyperparams)

    # two layer model
    def forward(self, z_input, t):

        z1_encode = self.encoder_layer_0(z_input, t)
        z2_encode = self.encoder_layer_1(z1_encode, t)

        z2_decode = z2_encode  # take the output of the encoder and feed it to the decoder

        z1_decode, x_decode_layer_1 = self.decoder_layer_1(z2_decode)
        z0_decode, x_decode_layer_0 = self.decoder_layer_0(z1_decode)

        return x_decode_layer_0, x_decode_layer_1, z0_decode

    # one layer model
    # def forward(self, z0):

    #     z1_encode = self.encoder_layer0(z0)

    #     z1_decode = z1_encode # take the output of the encoder and feed it to the decoder

    #     z0_decode = decoder_layer0(z1_decode)
