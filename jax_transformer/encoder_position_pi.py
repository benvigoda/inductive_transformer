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

from flax import linen as nn  # type: ignore
from typing import Callable
import jax.numpy as jnp  # type: ignore
from jax_transformer.helper_functions import (
    bound_activations,
    bound_weights
)
from jax.nn import logsumexp, log_softmax


class EncoderPositionPi(nn.Module):
    num_positions: int
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, rho):
        assert rho.shape == (self.num_positions, self.layer_width)

        weights = self.param(
            "weights", self.weight_init, (self.num_positions, self.layer_width)
        )
        #FIXME: removed log softmax
        # log_weights = log_softmax(weights, axis=0)
        log_weights = bound_weights(weights)

        # prob_weights = nn.relu(weights) + EPSILON
        # # NOTE: we decided to normalize the weights (it shouldn't matter)
        # prob_weights = custom_normalize(prob_weights, axis=0)

        # element-wise product of weight vector and token vector for each column in the layer
        # x = prob_weights * rho
        # make it an inner product by taking a sum along the token dimension
        # x = jnp.sum(x, axis=0, keepdims=True)

        x = logsumexp(log_weights + rho, axis=0, keepdims=True)

        assert x.shape == (1, self.layer_width)

        x = bound_activations(x)
        return x  # x is categorical


"""
Let vocab_size = 4, num_positions = 3, and layer_width = 2

The data is then a tensor that is size (num_positions=3, vocab size=4)

There is a Forney equals gates at each specific (position and word)

The left column has three pi_t's, each with 4 vocab words that can explain away the entire data
The right column also has this.

When the data says "small dog", then we want the left column to explain away the data

When the data says "big cat", then we want the right column to explain away the data

The entire layer_width must participate in explaining away the changing data.

We will need an open closed universe

... maybe we can do this in a simpler way that will not require the open closed universe:
in encoder_layer.py clone the data and send one clone straight up and one clone across
"""
