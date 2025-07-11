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

from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore
from jax.nn import log_softmax
from jax_transformer.helper_functions import bound_activations, bound_weights


class DecoderTokenPi(nn.Module):
    num_positions: int
    vocab_size: int
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, rho):
        # we expect rho to be already normalized categorical
        assert rho.shape == (self.num_positions, self.layer_width)

        weights = self.param(
            "weights",
            self.weight_init,
            (self.num_positions, self.vocab_size, self.layer_width),
        )
        #FIXME: removed log softmax
        log_weights = log_softmax(weights, axis=1)
        log_weights = bound_weights(log_weights)

        # FIXME: prob_weights = nn.relu(weights) + EPSILON

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the x value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        # FIXME: prob_weights = custom_normalize(prob_weights, axis=1)

        # rho = custom_normalize(rho, axis=1)  #FIXME: do we want this? We already do it in the decoder_position_pi
        # element-wise product of weight tensor and rho
        # FIXME: t = prob_weights * rho.reshape((self.num_positions, 1, self.layer_width))

        t = log_weights + rho.reshape(self.num_positions, 1, self.layer_width)
        assert t.shape == (self.num_positions, self.vocab_size, self.layer_width)

        t = bound_activations(t)
        return t
