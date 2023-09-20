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

        # at location 1 we receive a signal from
        # state 1 (in dim=0) from pi in the layer above at location 0 in dim=1
        # state 1 (in dim=0) from pi in the layer above at location 1 in dim=1

        return v #v is categorical