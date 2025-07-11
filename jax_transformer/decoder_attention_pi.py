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
from jax_transformer.helper_functions import (
    bound_weights,
    bound_activations
)
from jax.nn import log_softmax


class DecoderAttentionPi(nn.Module):
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, y):
        # we expect y to be already normalized categorical
        assert y.shape == (1, self.layer_width)

        weights = self.param(
            "weights", self.weight_init, (self.layer_width, self.layer_width)
        )
        # log_weights = log_softmax(weights, axis=1)
        # log_weights = log_softmax(weights, axis=0)
        #FIXME: removed log softmax
        log_weights = log_softmax(weights, axis=1)
        log_weights = log_softmax(log_weights, axis=0)
        log_weights = bound_weights(log_weights)

        # We want to interpret the weights as probabilities. To ensure they're all strictly between
        # 0 and 1, we pass them through a relu and then normalize.
        # prob_weights = nn.relu(weights)

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the y value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        # prob_weights = custom_normalize(prob_weights, axis=1)

        # and then since y comes in as categorical of size (1, layer_width)
        # y = custom_normalize(y, axis=1)

        # element-wise product of weight tensor and y
        v = log_weights + y

        # added this because like Bayes rule, Loeliger, chapter 2 of Ben's thesis, and
        # our deep belief in Bayes rule as foundational to concepts and thinking in transformers and humans
        # on the other hand, it works better if we don't normalize, because if the left hand pi_z is off
        # we want all it to send a message to all of its children to be OFF
        # v = custom_normalize(v, axis=0)

        assert v.shape == (self.layer_width, self.layer_width)
        # v[:, 0] = the outputs of the left hand attention pi
        # v[0, 0] = the straight down on the left output of the left hand attention pi
        # similarly, v[:, 1] = the outputs of the right hand attention pi

        v = bound_activations(v)
        return v
