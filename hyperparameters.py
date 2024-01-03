from typing import Optional, Dict
import torch  # type: ignore
from torch import Tensor  # type: ignore

PERTURBATION_TEST_WEIGHTS_TO_LEARN: Dict = {
    'encoder_attention': True,
    'encoder_token': False,
    'decoder_attention': True,
    'decoder_token': False,
}  # Set to True to manually set weights. Set to False to learn weights

STRONG = 1.  # Amplify the signal
WEAK = 1e-9  # Dampen the signal


class HyperParameters:
    def __init__(
            self,
            layer_width: int,
            vocab_size: int,
            num_positions: int,
            num_layers: int,
            unittest: bool = False,
            weight_test: bool = False,
            perturbation_test: bool = False,
    ):
        self.layer_width = layer_width
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.num_layers = num_layers
        self.unittest = unittest
        self.weight_test = weight_test
        self.perturbation_test = perturbation_test

        self.perturbation_test_encoder_attention = PERTURBATION_TEST_WEIGHTS_TO_LEARN['encoder_attention']
        self.perturbation_test_encoder_token = PERTURBATION_TEST_WEIGHTS_TO_LEARN['encoder_token']
        self.perturbation_test_decoder_attention = PERTURBATION_TEST_WEIGHTS_TO_LEARN['decoder_attention']
        self.perturbation_test_decoder_token = PERTURBATION_TEST_WEIGHTS_TO_LEARN['decoder_token']

        self.strong = STRONG  # Amplify the signal
        self.weak = WEAK  # Dampen the signal

        self.encoder_attention_pi_weights: Optional[Tensor] = None  # torch.ones(self.layer_width, self.layer_width)
        self.encoder_token_pi_weights: Optional[Tensor] = None  # torch.ones(self.vocab_size, self.layer_width)
        self.encoder_position_pi_weights: Optional[Tensor] = None  # torch.ones(self.num_positions, self.vocab_size, self.layer_width)
        self.decoder_attention_pi_weights: Optional[Tensor] = None  # torch.ones(self.layer_width, self.layer_width)
        self.decoder_token_pi_weights: Optional[Tensor] = None  # torch.ones(self.vocab_size, self.layer_width)
        self.decoder_position_pi_weights: Optional[Tensor] = None  # torch.ones(self.num_positions, self.vocab_size, self.layer_width)

        if self.weight_test:
            self.construct_weights()
        elif self.perturbation_test:
            self.construct_some_test_weights()

    def construct_some_test_weights(self):
        # Don't set the token weights, we can just let training take care of them
        print("Constructing some weights for perturbation test")

        if self.perturbation_test_encoder_token:
            self.encoder_token_pi_weights = torch.full((self.num_layers, self.vocab_size, self.layer_width), WEAK)
            self.encoder_token_pi_weights[0][0][0] = self.strong  # big in layer 0, left column
            self.encoder_token_pi_weights[0][3][1] = self.strong  # small in layer 0, right column
            self.encoder_token_pi_weights[1][1][0] = self.strong  # cat in layer 1, left column
            self.encoder_token_pi_weights[1][4][1] = self.strong  # dog in layer 1, right column
        if self.perturbation_test_decoder_token:
            self.decoder_token_pi_weights = torch.full((self.num_layers, self.vocab_size, self.layer_width), WEAK)
            self.decoder_token_pi_weights[0][0][0] = self.strong  # big in layer 0, left column
            self.decoder_token_pi_weights[0][3][1] = self.strong  # small in layer 0, right column
            self.decoder_token_pi_weights[1][1][0] = self.strong  # cat in layer 1, left column
            self.decoder_token_pi_weights[1][4][1] = self.strong  # dog in layer 1, right column
        if self.perturbation_test_encoder_attention:
            self.encoder_attention_pi_weights = torch.full((self.num_layers, self.layer_width, self.layer_width), WEAK)
            # Set the layer_0 attention weights to 0.5
            self.encoder_attention_pi_weights[0][0][0] = self.strong / 2
            self.encoder_attention_pi_weights[0][1][0] = self.strong / 2
            self.encoder_attention_pi_weights[0][0][1] = self.strong / 2
            self.encoder_attention_pi_weights[0][1][1] = self.strong / 2
            # Set the layer_1 straight connection weights to strong and leave the cross connection weights at 0
            self.encoder_attention_pi_weights[1][0][0] = self.strong
            self.encoder_attention_pi_weights[1][1][1] = self.strong
        if self.perturbation_test_decoder_attention:
            self.decoder_attention_pi_weights = torch.full((self.num_layers, self.layer_width, self.layer_width), WEAK)
            # Set the layer_0 attention weights to 0.5
            self.decoder_attention_pi_weights[0][0][0] = self.strong / 2
            self.decoder_attention_pi_weights[0][1][0] = self.strong / 2
            self.decoder_attention_pi_weights[0][0][1] = self.strong / 2
            self.decoder_attention_pi_weights[0][1][1] = self.strong / 2
            # Set the layer_1 straight connection weights to strong and leave the cross connection weights at 0
            self.decoder_attention_pi_weights[1][0][0] = self.strong
            self.decoder_attention_pi_weights[1][1][1] = self.strong

    def construct_weights(self):
        self.encoder_token_pi_weights = torch.full((self.num_layers, self.vocab_size, self.layer_width), WEAK)
        self.decoder_token_pi_weights = torch.full((self.num_layers, self.vocab_size, self.layer_width), WEAK)
        self.encoder_attention_pi_weights = torch.full((self.num_layers, self.layer_width, self.layer_width), WEAK)
        self.decoder_attention_pi_weights = torch.full((self.num_layers, self.layer_width, self.layer_width), WEAK)

        self.encoder_token_pi_weights[1][0][0] = self.strong  # cat in layer 0, left column
        self.encoder_token_pi_weights[1][3][1] = self.strong  # dog in layer 0, right column
        self.encoder_token_pi_weights[0][1][0] = self.strong  # big in layer 1, left column
        self.encoder_token_pi_weights[0][4][1] = self.strong  # small in layer 1, right column

        self.decoder_token_pi_weights[1][0][0] = self.strong  # cat in layer 0, left column
        self.decoder_token_pi_weights[1][3][1] = self.strong  # dog in layer 0, right column
        self.decoder_token_pi_weights[0][1][0] = self.strong  # big in layer 1, left column
        self.decoder_token_pi_weights[0][4][1] = self.strong  # small in layer 1, right column

        # Set the layer_0 attention weights to 0.5
        self.encoder_attention_pi_weights[0][0][0] = self.strong / 2
        self.encoder_attention_pi_weights[0][1][0] = self.strong / 2
        self.encoder_attention_pi_weights[0][0][1] = self.strong / 2
        self.encoder_attention_pi_weights[0][1][1] = self.strong / 2
        # Set the layer_1 straight connection weights to strong and leave the cross connection weights at 0
        self.encoder_attention_pi_weights[1][0][0] = self.strong
        self.encoder_attention_pi_weights[1][1][1] = self.strong

        # Set the layer_0 attention weights to 0.5
        self.decoder_attention_pi_weights[0][0][0] = self.strong / 2
        self.decoder_attention_pi_weights[0][1][0] = self.strong / 2
        self.decoder_attention_pi_weights[0][0][1] = self.strong / 2
        self.decoder_attention_pi_weights[0][1][1] = self.strong / 2
        # Set the layer_1 straight connection weights to strong and leave the cross connection weights at 0
        self.decoder_attention_pi_weights[1][0][0] = self.strong
        self.decoder_attention_pi_weights[1][1][1] = self.strong
