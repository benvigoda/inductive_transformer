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
from jax_transformer.helper_functions import custom_normalize, bound_activations


@dataclass
class DecoderAnd:
    layer_width: int

    # Toggle this to use the encoder message. In theory this should be True, but there could be an
    # error in there and also, it should be simpler without the encoder message.
    use_encoder_message: bool = True

    def __call__(self, z, x_encoder, y_encoder):
        assert x_encoder.shape == (2, self.layer_width)
        assert y_encoder.shape == (2, self.layer_width)
        assert z.shape == (2, self.layer_width)

        # OLD AND:
        # y0_z0 = y_encoder[0] * z[0]
        # x_0 = y0_z0 + y_encoder[1] * z[0]
        # x_1 = y0_z0 + y_encoder[1] * z[1]

        # x0_z0 = x_encoder[0] * z[0]
        # y_0 = x0_z0 + x_encoder[1] * z[0]
        # y_1 = x0_z0 + x_encoder[1] * z[1]

        # NEW EQUAL
        # x_0 = y_encoder[0] * z[0]
        # x_1 = y_encoder[1] * z[1]
        # x = jnp.stack([x_0, x_1], axis=0)
        x = y_encoder + z

        # y_0 = x_encoder[0] * z[0]
        # y_1 = x_encoder[1] * z[1]
        # y = jnp.stack([y_0, y_1], axis=0)
        y = x_encoder + z

        # import pdb; pdb.set_trace()
        # x = custom_normalize(x, axis=0)
        # y = custom_normalize(y, axis=0)

        # x = bound_activations(x)
        # y = bound_activations(y)
        return x, y  # Bernoullis
