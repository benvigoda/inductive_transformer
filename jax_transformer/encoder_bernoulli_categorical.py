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

from dataclasses import dataclass

from jax_transformer.helper_functions import custom_normalize, EPSILON, bound_activations


@dataclass
class EncoderBernoulliCategorical:
    def __call__(self, u):
        # v = u[1] / (u[0] + EPSILON)
        # if u is properly normalized then we should not need to divide by zero
        v = u[1]

        v = bound_activations(v)
        return v
