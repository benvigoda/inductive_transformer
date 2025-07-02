# Copyright 2024 Your Name
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
from helper_functions import bound_activations


@dataclass
class DecoderCategoricalBernoulli:
    layer_width: int

    def __call__(self, v):
        assert v.shape == (self.layer_width, self.layer_width)

        # The probability of a bernoulli variable being True is the same as the probability of the
        # corresponding categorical state.
        u_1 = v

        # The probability of a bernoulli variable being False is 1 - the probability of it being True.
        # pu_0 = 1.0 - pu_1
        # u_0 = log(1 - pu_1)
        # u_0 = log(1 - exp(u_1))
        u_0 = jnp.log1p(-jnp.exp(u_1))

        u = jnp.stack([u_0, u_1], axis=0)

        assert u.shape == (2, self.layer_width, self.layer_width)

        u = bound_activations(u)
        return u
