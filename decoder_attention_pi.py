import torch  # type: ignore
from torch import nn  # type: ignore


class DecoderAttentionPi(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderAttentionPi, self).__init__()
        self.hyperparams = hyperparams
        self.layer_width = self.hyperparams.layer_width
        self.active_layer = active_layer
        if hyperparams.decoder_attention_pi_weights is not None:
            self.weights = hyperparams.decoder_attention_pi_weights[active_layer]
        else:
            self.weights = nn.Parameter(torch.ones(self.layer_width, self.layer_width), requires_grad=True)
            nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()

    def forward(self, y):

        # we expect y to be already normalized categorical

        prob_weights = self.relu(self.weights) + 1e-9

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the y value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        prob_weights = torch.normalize(prob_weights, p=1, dim=0)

        # and then since y comes in as categorical of size (1, layer_width)
        assert y.shape == (1, self.layer_width)
        y = torch.normalize(y, dim=1)
        # we want to stack x in dim = 0
        y_stacked = torch.stack([y for lw in range(self.layer_width)], dim=0)

        # element-wise product of weight tensor and y_stacked
        v = prob_weights * y_stacked
        assert v.shape == (self.layer_width, self.layer_width)

        return v

        '''
        # decoder_categorical_bernoulli.py
        # u = f(v)
        v is size (layer_width, layer_width)
        # Note that "above" is incoming in the decoder while "below" is outgoing
        dim = 1 indexes across the layer, above_lw which is short for "above layer width"
        dim = 0 indexes the choice that a given attention pi is making
        we should think of this dim=0 above choice as choosing one of the concepts in the layer below
        therefore dim=0 indexes the layer below.  we'll call that index, below_lw "below layer width"
        below_lw = above_choice

        let's think in terms of below_lw
        at below_lw=0, we receive a signal from attention_pi's at alw = 0 and alw = 1
        That is two signals


        # decoder_universe.py





        # 1.
        # state 0 (in dim=0) from pi in the layer above at location 0 in dim=1
        # in this u the size is (bernoulli state, lw for layer below, lw for layer above)
        u[1][0][0] = v[0][0] # remember v is size = (1, layer_width)
        # the zero state below is the sum of the other categorical values from the layer above
        u[0][0][0] = v[0][1]

        # we converted to Bernoullis and those become the two parents of
        # the AND at location 0

        # 2.
        # state 0 (in dim=0) from pi in the layer above at location 1 in dim=1
        # in this u the size is (bernoulli state, lw for layer below, lw for layer above)
        u[1][0][1] = v[0][1]
        # the zero state below is the sum of the other categorical states from the later above
        u[0][0][1] = v[1][1]




        # at location 1 we receive a signal from
        # state 1 (in dim=0) from pi in the layer above at location 0 in dim=1
        # state 1 (in dim=0) from pi in the layer above at location 1 in dim=1



        # we will convert both of these to coins and those are the two parents of
        # the AND at location 1

        # code the Bernoulli to categorical here and then transfer it to its own file



        u[1][0][1] = v[0][1]


        # code the decoder universe here and also transfer it
        '''
