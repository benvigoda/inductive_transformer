from typing import Optional, Dict
import torch  # type: ignore
from torch import Tensor  # type: ignore

PERTURBATION_TEST_WEIGHTS_TO_LEARN: Dict = {
    'encoder_decision': True,
    'encoder_word': True,
    'decoder_allsum': True,
    'decoder_word': True,
}

STRONG = 1.  # Amplify the signal
WEAK = 1e-9  # Dampen the signal


class HyperParameters:
    def __init__(
            self,
            layer_width: int,
            vocab_size: int,
            num_layers: int,
            unittest: bool = False,
            weight_test: bool = False,
            perturbation_test: bool = False,
    ):
        self.layer_width = layer_width
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.unittest = unittest
        self.weight_test = weight_test
        self.perturbation_test = perturbation_test

        self.perturbation_test_encoder_decision = PERTURBATION_TEST_WEIGHTS_TO_LEARN['encoder_decision']
        self.perturbation_test_encoder_word = PERTURBATION_TEST_WEIGHTS_TO_LEARN['encoder_word']
        self.perturbation_test_decoder_all = PERTURBATION_TEST_WEIGHTS_TO_LEARN['decoder_allsum']
        self.perturbation_test_decoder_word = PERTURBATION_TEST_WEIGHTS_TO_LEARN['decoder_word']

        self.strong = STRONG  # Amplify the signal
        self.weak = WEAK  # Dampen the signal

        self.encoder_attention_pi_weights: Optional[Tensor] = None  # torch.ones(self.layer_width, self.layer_width)
        self.encoder_token_pi_weights: Optional[Tensor] = None  # torch.ones(self.vocab_size, self.layer_width)
        self.decoder_attention_pi_weights: Optional[Tensor] = None  # torch.ones(self.layer_width, self.layer_width)
        self.decoder_token_pi_weights: Optional[Tensor] = None  # torch.ones(self.vocab_size, self.layer_width)

        if self.weight_test:
            self.construct_weights()
        elif self.perturbation_test:
            self.construct_some_test_weights()

    def construct_some_test_weights(self):
        # Don't set the word weights, we can just let training take care of them
        print("Constructing some weights for perturbation test")

        # self.weak decision cross-connections, so we send signals straight up and down the columns
        if self.perturbation_test_encoder_decision:
            self.encoder_attention_pi_weights = torch.full((self.num_layers, self.layer_width, self.layer_width), self.strong)
            for n in range(self.num_layers):
                for i in range(self.layer_width):
                    for lw in range(self.layer_width):
                        if i != lw:
                            self.encoder_attention_pi_weights[n][i][lw] = self.weak

        if self.perturbation_test_decoder_all:
            self.decoder_attention_pi_weights = torch.full((self.num_layers, self.layer_width, self.layer_width), self.strong)
            for n in range(self.num_layers):
                for i in range(self.layer_width):
                    for lw in range(self.layer_width):
                        if i != lw:
                            self.decoder_attention_pi_weights[n][i][lw] = self.weak

        if self.perturbation_test_encoder_word:
            self.encoder_token_pi_weights = torch.full((self.num_layers, self.vocab_size, self.layer_width), self.weak)
            self.encoder_token_pi_weights[0][1][0] = self.strong  # dog in layer 0, left column
            self.encoder_token_pi_weights[0][4][1] = self.strong  # cat in layer 0, right column
            if self.num_layers >= 2:
                self.encoder_token_pi_weights[1][0][0] = self.strong  # the in layer 1, left column
                self.encoder_token_pi_weights[1][3][1] = self.strong  # a in layer 1, right column
            if self.num_layers == 3:
                self.encoder_token_pi_weights[2][5][0] = self.strong  # always light up for PADDING
                self.encoder_token_pi_weights[2][5][1] = self.strong  # always light up for PADDING

        if self.perturbation_test_decoder_word:
            self.decoder_token_pi_weights = torch.full((self.num_layers, self.layer_width, self.vocab_size), self.weak)
            self.decoder_token_pi_weights[0][0][1] = self.strong
            self.decoder_token_pi_weights[0][1][4] = self.strong
            if self.num_layers >= 2:
                self.decoder_token_pi_weights[1][0][0] = self.strong
                self.decoder_token_pi_weights[1][1][3] = self.strong

            if self.num_layers == 3:
                self.decoder_token_pi_weights[2][0][5] = self.strong
                self.decoder_token_pi_weights[2][1][5] = self.strong

    def construct_weights(self):
        '''
        EXAMPLE:

        2 sentences "the dog. a cat." in the `text_training.txt` file
        vocab: ['the', 'dog', '.', 'cat', 'a', '<PADDING>'],

        sentence 1 "the dog":
        layer 1 left = "the" = (0., -5., -5., -5., -5., -5.)
        layer 0 left  = "dog" = (-5, 0., -5., -5., -5., -5.)
        layer 1 right = NA = (-5., -5., -5., -5., -5., -5.)
        layer 0 right = NA = (-5., -5., -5., -5., -5., -5.)

        sentence 2 "a cat":
        layer 1 left = NA = (-5., -5., -5., -5., -5., -5.)
        layer 0 left = NA = (-5., -5., -5., -5., -5., -5.)
        layer 1 right = "a" = (-5, -5., -5., -5., 0., -5.)
        layer 0 right  = "cat" = (-5, -5., -5., 0., -5., -5.)
        '''
        self.encoder_token_pi_weights = torch.ones(self.vocab_size, self.layer_width)
        self.decoder_token_pi_weights = torch.ones(self.vocab_size, self.layer_width)
        self.encoder_attention_pi_weights = torch.ones(self.layer_width, self.layer_width)
        self.decoder_attention_pi_weights = torch.ones(self.layer_width, self.layer_width)
        # self.encoder_token_pi_weights = torch.full((self.num_layers, self.vocab_size, self.layer_width), self.weak)
        # self.decoder_token_pi_weights = torch.full((self.num_layers, self.layer_width, self.vocab_size), self.weak)

        # self.encoder_token_pi_weights[0][1][0] = self.strong  # dog in layer 0, left column
        # self.encoder_token_pi_weights[0][4][1] = self.strong  # cat in layer 0, right column

        # self.decoder_token_pi_weights[0][0][1] = self.strong
        # self.decoder_token_pi_weights[0][1][4] = self.strong

        # if self.num_layers >= 2:
        #     self.encoder_token_pi_weights[1][0][0] = self.strong  # the in layer 1, left column
        #     self.encoder_token_pi_weights[1][3][1] = self.strong  # a in layer 1, right column

        #     self.decoder_token_pi_weights[1][0][0] = self.strong
        #     self.decoder_token_pi_weights[1][1][3] = self.strong

        # if self.num_layers == 3:
        #     self.encoder_token_pi_weights[2][5][0] = self.strong  # always light up for PADDING
        #     self.encoder_token_pi_weights[2][5][1] = self.strong  # always light up for PADDING

        #     self.decoder_token_pi_weights[2][0][5] = self.strong
        #     self.decoder_token_pi_weights[2][1][5] = self.strong

        # # self.weak decision cross-connections, so we send signals straight up and down the columns
        # self.encoder_attention_pi_weights = torch.full((self.num_layers, self.layer_width, self.layer_width), self.strong)
        # self.decoder_attention_pi_weights = torch.full((self.num_layers, self.layer_width, self.layer_width), self.strong)
        # for n in range(self.num_layers):
        #     for i in range(self.layer_width):
        #         for lw in range(self.layer_width):
        #             if i != lw:
        #                 self.encoder_attention_pi_weights[n][i][lw] = self.weak
        #                 self.decoder_attention_pi_weights[n][i][lw] = self.weak
