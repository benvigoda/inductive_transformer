from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from jax_transformer.helper_functions import custom_normalize


@dataclass
class EncoderAnd:

    def __call__(self, x, y):

        z_1 = x[1] * y[1]
        z_0 = x[0] * y[1] + x[1] * y[0] + x[0] * y[0]
        z = jnp.stack([z_0, z_1])

        z = custom_normalize(z, axis=0)
        return z
