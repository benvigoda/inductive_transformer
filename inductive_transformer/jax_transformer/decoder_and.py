from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from inductive_transformer.jax_transformer.helper_functions import custom_normalize


@dataclass
class DecoderAnd:
    layer_width: int

    # Toggle this to use the encoder message. In theory this should be True, but there could be an
    # error in there and also, it should be simpler without the encoder message.
    use_encoder_message: bool = True

    def __call__(self, z, x_encoder, y_encoder):


        assert z.shape == (2, self.layer_width)

        if self.use_encoder_message and x_encoder is not None and y_encoder is not None:
            assert x_encoder.shape == (2, self.layer_width)
            assert y_encoder.shape == (2, self.layer_width)
            assert z.shape == (2, self.layer_width)

            # OLD AND:
            # y0_z0 = y_encoder[0] * z[0]
            # x_0 = y0_z0 + y_encoder[1] * z[0]
            # x_1 = y0_z0 + y_encoder[1] * z[1]
            # NEW EQUAL
            # x_0 = y_encoder[0] * z[0]
            # x_1 = y_encoder[1] * z[1]
            # x = jnp.stack([x_0, x_1], axis=0)
            x = y_encoder * z

            # OLD AND:
            # x0_z0 = x_encoder[0] * z[0]
            # y_0 = x0_z0 + x_encoder[1] * z[0]
            # y_1 = x0_z0 + x_encoder[1] * z[1]
            # NEW EQUAL
            # y_0 = x_encoder[0] * z[0]
            # y_1 = x_encoder[1] * z[1]
            # y = jnp.stack([y_0, y_1], axis=0)
            y = x_encoder * z

        # Note we should not use this
        # Not sure it is even up to date
        else:
            raise NotImplementedError

        # import pdb; pdb.set_trace()
        x = custom_normalize(x, axis=0)
        y = custom_normalize(y, axis=0)

        return x, y   # Bernoullis
