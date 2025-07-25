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
import jax.numpy as jnp  # type: ignore
from typing import Callable
from jax_transformer.helper_functions import (
    bound_activations,
    bound_weights
)
from jax.nn import logsumexp, log_softmax
from helper_functions import custom_normalize


class EncoderAttentionPi(nn.Module):
    vocab_size: int
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, v):
        assert v.shape == (self.layer_width, self.layer_width)
        # we expect v to be already normalized categorical
        weights = self.param(
            "weights", self.weight_init, (self.layer_width, self.layer_width)
        )

        #FIXME: removed log softmax
        # log_weights = log_softmax(weights, axis=0)  # given the same seed, gets to lower loss, faster
        log_weights = bound_weights(weights)
        # log_weights = log_softmax(weights, axis=1)

        # prob_weights = nn.relu(weights) + EPSILON
        # prob_weights = custom_normalize(prob_weights, axis=1)
        # # in the future we may want to normalize v here for good measure
        # v = custom_normalize(v, axis=1)
        # # element-wise product of weight vector and token vector for each column in the layer
        # y = prob_weights * v
        # # make it an inner product by taking a sum along the choice dimension
        # y = jnp.sum(y, axis=0, keepdims=True)  # after summing it is size = (1, layer_width)

        y = logsumexp(log_weights + v, axis=0, keepdims=True)
        assert y.shape == (1, self.layer_width)

        # we had to remove this since otherwise, y_categorical would have 0.5's instead of 1's,
        # when it is certain on both values of the layer_width index:
        # y = custom_normalize(y, axis=1)

        # y = log_softmax(y, axis=1)
        y = bound_activations(y)
        return y  # y is categorical
