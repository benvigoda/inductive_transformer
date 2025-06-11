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

from jax_transformer.helper_functions import custom_normalize, bound_activations


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

        z = bound_activations(z)
        return z
