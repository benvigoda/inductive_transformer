from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from inductive_transformer.jax_transformer.helper_functions import custom_normalize


@dataclass
class DecoderUniverse:
    layer_width: int

    def __call__(self, u):
        assert u.shape == (2, self.layer_width, self.layer_width)

        z = jnp.sum(u, axis=1)
        """
        # u[heads/tails][below_lw?][above_lw?]
        # left

        z[0][0] = u[0][0][0] * u[0][0][1]  # all the others
        z[1][0] = u[1][0][0] * u[1][0][1] + u[1][0][0] * u[0][0][1] + u[0][0][0] * u[1][0][1]
        # right
        z[0][1] = u[0][1][0] * u[0][1][1]
        z[1][1] = u[1][1][0] * u[1][1][1] + u[1][1][0] * u[0][1][1] + u[0][1][0] * u[1][1][1]

        # z = nn.functional.normalize(z, p=1, dim=0)
        """
        z = custom_normalize(z, axis=0)
        return z
