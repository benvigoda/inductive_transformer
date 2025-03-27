from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from jax_transformer.helper_functions import custom_normalize


@dataclass
class EncoderAnd:
    def __call__(self, x, y):

        # OLD SOFTAND:
        # z_1 = x[1] * y[1]
        # z_0 = x[0] * y[1] + x[1] * y[0] + x[0] * y[0]
        
        # NEW SOFTEQUAL
        z_1 = x[1] + y[1]
        z_0 = x[0] + y[0]

        # FIXME: since logsumexp of the activations coming into this gate are shifting the values by a constant
        # the incoming values x[1] and y[1] have been arbitrarily normalized

        z = jnp.stack([z_0, z_1])
        z = custom_normalize(z, axis=0)

        return z

