import torch  # type: ignore
import jax.numpy as jnp

from helper_functions import custom_normalize

@dataclass
class EncoderAnd:
    layer_width: int

    def __call__(self, x, y):
        self.x = x
        self.y = y
        z = torch.empty((2, self.layer_width), device=x.device)
        z_1 = x[1] * y[1]
        z_0 = x[0] * y[1] + x[1] * y[0] + x[0] * y[0]
        z = jnp.stack(z_0, z_1)

        z = custom_normalize(z, dim=0)
        return z
