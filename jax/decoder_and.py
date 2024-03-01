from dataclasses import dataclass
from helper_functions import custom_normalize
import jax.numpy as jnp


@dataclass
class DecoderAnd:
    layer_width: int

    # Toggle this to use the encoder message. In theory this should be True, but there could be an
    # error in there and also, it should be simpler without the encoder message.
    use_encoder_message: bool = True

    def __call__(self, z, x_encoder, y_encoder):
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

        assert z.shape == (2, self.layer_width)

        if self.use_encoder_message and x_encoder is not None and y_encoder is not None:
            assert x_encoder.shape == (2, self.layer_width)
            assert y_encoder.shape == (2, self.layer_width)

            # torch
            # x[0] = y_encoder[0] * z[0] + y_encoder[1] * z[0]
            # x[1] = y_encoder[0] * z[0] + y_encoder[1] * z[1]

            y0_z0 = y_encoder[0] * z[0]
            x = jnp.stack([
                y0_z0 + y_encoder[1] * z[0],
                y0_z0 + y_encoder[1] * z[1],
            ])

            # torch
            # y[0] = x_encoder[0] * z[0] + x_encoder[1] * z[0]
            # y[1] = x_encoder[0] * z[0] + x_encoder[1] * z[1]

            x0_z0 = x_encoder[0] * z[0]
            y = jnp.stack([
                x0_z0 + x_encoder[1] * z[0],
                x0_z0 + x_encoder[1] * z[1],
            ])

        else:
            # torch
            # x[0] = z[0] + z[0]
            # x[1] = z[0] + z[1]

            x = jnp.stack([
                z[0] + z[0],
                z[0] + z[1],
            ])

            # torch
            # y[0] = z[0] + z[0]
            # y[1] = z[0] + z[1]

            y = jnp.stack([
                z[0] + z[0],
                z[0] + z[1],
            ])

        x = custom_normalize(x, axis=0)
        y = custom_normalize(y, axis=0)

        return x, y   # Bernoullis
