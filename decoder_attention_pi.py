import torch  # type: ignore
from torch import nn  # type: ignore


class DecoderAttentionPi(nn.Module):

    def __init__(self, hyperparams):
        super(DecoderAttentionPi, self).__init__()
        self.hyperparams = hyperparams
        self.layer_width = self.hyperparams.layer_width

        self.weights = nn.Parameter(torch.ones(self.layer_width, self.layer_width), requires_grad=True)
        nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()

    def forward(self, y):
        # we expect y to be already normalized categorical

        prob_weights = self.relu(self.weights) + 1e-9

        # element-wise product of weight vector and token vector for each column in the layer
        for y in range(self.layer_width):
            y_stacked = torch.stack([y_stacked], y, dim=0)
        v = prob_weights * y_stacked

        v = torch.normalize(v, p=1, dim=0)

        # ANDs in the layer below with layer width indexed by dim=0

        # at location 0 we receive a signal from
        # state 0 (in dim=0) from pi in the layer above at location 0 in dim=1
        # state 0 (in dim=0) from pi in the layer above at location 1 in dim=1

        # we will convert both of these to coins and those are the two parents of
        # the AND at location 0

        # at location 1 we receive a signal from
        # state 1 (in dim=0) from pi in the layer above at location 0 in dim=1
        # state 1 (in dim=0) from pi in the layer above at location 1 in dim=1

        # we will convert both of these to coins and those are the two parents of
        # the AND at location 1

        return v  # v is categorical

        """
        I lost a little steam here
        I am figuring out the final flow from decoder
        attention pi to coins to closed universe
        kind of a little tricky

        Sounds good :) I'm going to head to bed as well.
        yeah probably need to sleep now
        want to commit?
        I did (slacked you a few messages and what I think we should do next)
        thank you for making that list in slack
        sounds great
        okay sleep time!
        Good night!

        just figured it out!
        okay now actually going to sleep :-)
        can you make sure lines 34 and 41 get saved for tomorrow?
        I'll just put them in slack

        Sure. I'll make one more commit :). nah go to sleep
        we're good

        """
