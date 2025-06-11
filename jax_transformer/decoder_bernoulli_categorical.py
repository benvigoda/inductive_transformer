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
class DecoderBernoulliCategorical:
    layer_width: int

    def __call__(self, bernoulli):
        # bernoulli is size (2, layer_width)
        assert bernoulli.shape == (2, self.layer_width)

        # Tried removing denominator
        bernoulli = custom_normalize(bernoulli, axis=0)  # Should not be necessary (we already normalize at the end of the decoder_and)
        categorical = bernoulli[1]
        categorical = categorical.reshape((1, self.layer_width))

        # Removed the layer norm

        categorical = bound_activations(categorical)
        return categorical
