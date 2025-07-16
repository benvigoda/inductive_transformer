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
from helper_functions import custom_normalize

class EncoderTokenPi(nn.Module):
    num_positions: int
    layer_width: int
    vocab_size: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, t):
        assert t.shape == (self.num_positions, self.vocab_size, self.layer_width)
        # we expect t to be already normalized
        t = bound_activations(t)
        
        weights = self.param(
            "weights",
            self.weight_init,
            (self.num_positions, self.vocab_size, self.layer_width),
        )
        # FIXME: we will want to reove this:
        # log_weights = log_softmax(weights, axis=1)
        log_weights = bound_weights(weights)

        # # in the probability domain:
        # element-wise product of weight vector and token vector for each column in the layer
        # this is an element-wise product, no axis should be specified
        # rho = logprob_weights * t
        rho = log_weights + t

        # the inner product then requires a sum along the token dimension
        rho = logsumexp(rho, axis=1)

        # after summing it is size = (num_positions, layer_width)
        # normalize rho over the position dimension
        # rho = custom_normalize(rho, axis=0)

        # rho = log_softmax(rho, axis=0)

        rho = bound_activations(rho)
        return rho  # rho is categorical
