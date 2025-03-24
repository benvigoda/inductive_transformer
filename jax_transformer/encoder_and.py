from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from jax_transformer.helper_functions import custom_normalize


@dataclass
class EncoderAnd:
    def __call__(self, x, y):

        x_norm_const = x[0] + x[1]
        x[1] = x[1]/x_norm_const
        x[0] = x[0]/x_norm_const

        y_norm_const = x[0] + x[1]
        y[1] = y[1]/y_norm_const
        y[0] = y[0]/y_norm_const

        x_norm_const = min(x[0], x[1])
        x[0] = -20
        x[1] = -10
        x_norm_const = -20
        #x[0] --> 0
        #x[1] --> 10
        #p[0]

        x[1] = x[1] - x_norm_const


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

