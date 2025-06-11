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
import jax.numpy as jnp  # type: ignore
from typing import Callable
from jax_transformer.helper_functions import bound_activations, bound_weights
from jax.nn import logsumexp, log_softmax


class EncoderTokenPi(nn.Module):
    num_positions: int
    layer_width: int
    vocab_size: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, t):
        assert t.shape == (self.num_positions, self.vocab_size, self.layer_width)
        # we expect t to be already normalized

        weights = self.param(
            "weights",
            self.weight_init,
            (self.num_positions, self.vocab_size, self.layer_width),
        )
        log_weights = log_softmax(weights, axis=0)
        log_weights = bound_weights(log_weights)

        # FIXME: Is this all getting properly normalized?
        # logprob_weights = nn.relu(weights) + EPSILON

        # # element-wise product of weight vector and token vector for each column in the layer
        # rho = logprob_weights * t

        # # make it an inner product by taking a sum along the token dimension
        # rho = jnp.sum(rho, axis=1)  # after summing it is size = (num_positions, layer_width)

        # this replaces a prob domain element-wise product followed by sum on the axis=1
        rho = logsumexp(log_weights + t, axis=1)

        rho = bound_activations(rho)

        return rho  # rho is categorical
