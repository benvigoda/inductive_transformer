import torch  # type: ignore
from torch import nn  # type: ignore
from helper_functions import custom_normalize


class DecoderAnd(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderAnd, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

        self.y = None
        self.x = None

    def forward(self, z, x_encoder, y_encoder):
        '''
        x y z prob
        0 0 0   1
        0 0 1   0
        0 1 0   1
        0 1 1   0
        1 0 0   1
        1 0 1   0
        1 1 0   0
        1 1 1   1

        prob = 1 states when y = 1
        0 1 0   1
        1 1 1   1

        prob = 1 states when y = 0
        0 0 0   1
        1 0 0   1
        '''

        # left
        # y[1][0] = x[0][0]*z[0][0] + x[1][0]*z[1][0]
        # y[0][0] = x[0][0]*z[0][0] + x[1][0]*z[0][0]

        # x[1][0] = y[0][0]*z[0][0] + y[1][0]*z[1][0]
        # x[0][0] = y[0][0]*z[0][0] + y[1][0]*z[0][0]

        # right
        # y[1][1] = x[0][1]*z[0][1] + x[1][1]*z[1][1]
        # y[0][1] = x[0][1]*z[0][1] + x[1][1]*z[0][1]

        # x[1][1] = y[0][1]*z[0][1] + y[1][1]*z[1][1]
        # x[0][1] = y[0][1]*z[0][1] + y[1][1]*z[0][1]

        # --------------

        # left
        # y[1][0] = 0.5*z[0][0] + 0.5*z[1][0]
        # y[0][0] = 0.5*z[0][0] + 0.5*z[0][0]

        # x[1][0] = 0.5*z[0][0] + 0.5*z[1][0]
        # x[0][0] = 0.5*z[0][0] + 0.5*z[0][0]

        # right
        # y[1][1] = 0.5*z[0][1] + 0.5*z[1][1]
        # y[0][1] = 0.5*z[0][1] + 0.5*z[0][1]

        # x[1][1] = 0.5*z[0][1] + 0.5*z[1][1]
        # x[0][1] = 0.5*z[0][1] + 0.5*z[0][1]

        # --------------

        x = torch.empty((2, 2))
        y = torch.empty((2, 2))
        use_encoder_message = True  # Toggle this to use the encoder message
        # In theory this should be True, but there could be an error in there
        # and also, it should be simpler without the encoder message.
        if use_encoder_message and x_encoder is not None and y_encoder is not None:
            # left
            y[1][0] = x_encoder[0][0]*z[0][0] + x_encoder[1][0]*z[1][0]
            y[0][0] = x_encoder[0][0]*z[0][0] + x_encoder[1][0]*z[0][0]

            x[1][0] = y_encoder[0][0]*z[0][0] + y_encoder[1][0]*z[1][0]
            x[0][0] = y_encoder[0][0]*z[0][0] + y_encoder[1][0]*z[0][0]

            # right
            y[1][1] = x_encoder[0][1]*z[0][1] + x_encoder[1][1]*z[1][1]
            y[0][1] = x_encoder[0][1]*z[0][1] + x_encoder[1][1]*z[0][1]

            x[1][1] = y_encoder[0][1]*z[0][1] + y_encoder[1][1]*z[1][1]
            x[0][1] = y_encoder[0][1]*z[0][1] + y_encoder[1][1]*z[0][1]
        else:
            # left
            y[1][0] = z[0][0] + z[1][0]
            y[0][0] = z[0][0] + z[0][0]

            x[1][0] = z[0][0] + z[1][0]
            x[0][0] = z[0][0] + z[0][0]

            # right
            y[1][1] = z[0][1] + z[1][1]
            y[0][1] = z[0][1] + z[0][1]

            x[1][1] = z[0][1] + z[1][1]
            x[0][1] = z[0][1] + z[0][1]

        # y = nn.functional.normalize(y, p=1, dim=0)
        # x = nn.functional.normalize(x, p=1, dim=0)
        x = custom_normalize(x, dim=0)
        y = custom_normalize(y, dim=0)

        self.y = y
        self.x = x

        return x, y   # Bernoullis
