# Copyright 2025 Ben Vigoda, Thomas Rochais, and Erik Strand
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from jax_transformer.helper_functions import bound_activations


@dataclass
class DecoderUniverse:
    layer_width: int

    def __call__(self, u):
        assert u.shape == (2, self.layer_width, self.layer_width)
        """
        what is u coming in?

        it's v as the HEADS of the Bernoulli
        # v[:, 0] = the outputs of the left hand attention pi
        # v[0, 0] = the straight down on the left output of the left hand attention pi
        # similarly, v[:, 1] = the outputs of the right hand attention pi

        stacked on axis=0 with (1-v) as the TAILS of the Bernoulli

        resulting in shape:
        assert u.shape == (2, left and right edges coming out of the an attention unit, left and right attention units)

        How would we consume this?  To get the two parents of the left-side universe we need:
        parent_a = v[0, 0] the straight down edge of the left-side attention unit
        parent_b v[0, 1] the left-going edge of the right-side attention unit
        and we want TAILS of both of these, like this:

        z[0][0] = u[0][0][0] * u[0][0][1]
        z[1][0] = 1 - z[0][0]
        z[0][1] = u[0][1][0] * u[0][1][1]
        z[1][1] = 1 - z[0][1]
        """

        # more general form of this:
        # z_0 = jnp.sum(u[0], axis=-1)
        # z_1 = 1 - z_0
        # z = jnp.stack([z_0, z_1], axis=0)

        z_0 = jnp.sum(u[0], axis=-1)
        z_1 = jnp.log1p(-jnp.exp(z_0))
        z = jnp.stack([z_0, z_1], axis=0)

        # z = custom_normalize(z, axis=0)  # Has to be unecessary Otherwise we would not be allowed to do z_1 = 1 - z_0
        assert z.shape == (2, self.layer_width)

        z = bound_activations(z)
        return z
