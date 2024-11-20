from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from inductive_transformer.jax_transformer.helper_functions import custom_normalize


@dataclass
class EncoderAnd:
    def __call__(self, x, y):
        # OLD AND:
        # z_1 = x[1] * y[1]
        # z_0 = x[0] * y[1] + x[1] * y[0] + x[0] * y[0]
        # NEW EQUAL
        z_1 = x[1] * y[1]
        z_0 = x[0] * y[0]
        z = jnp.stack([z_0, z_1])

        z = custom_normalize(z, axis=0)
        # z_nan = jnp.isnan(z).any()
        # try:
        #     if z_nan.val[0]:
        #         print("nan in z at encoder_and")
        # except:
        #     if z_nan:
        #         print("nan in z at encoder_and")
        return z
