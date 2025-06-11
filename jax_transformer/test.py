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

import jax.numpy as jnp  # type: ignore
import numpy as np

EPSILON = 1e-15
PROBABLE = 1 - EPSILON
IMPROBABLE = EPSILON


def custom_normalize(tensor: jnp.ndarray, axis=0, default_constant=0.5) -> jnp.ndarray:
    """
    axis is the dimension on which to normalize
    default_constant is the value to use when the sum is zero
    """
    # Compute the sum along axis=axis and keepdims=True to maintain the dimensions for broadcasting
    sum_tensor = jnp.sum(tensor, axis=axis, keepdims=True)

    # Create a mask where the sum is zero
    mask = sum_tensor == 0

    # Replace zero sums with ones to avoid division by zero and then divide
    result = tensor / jnp.where(mask, jnp.ones_like(sum_tensor), sum_tensor)

    # Where the sum was zero, replace with the constant C
    result = jnp.where(mask, jnp.full_like(result, fill_value=default_constant), result)

    return result


z = np.array([[1, 0], [0, 1]])
y_encoder = np.array([[0.5] * 2] * 2)
x = y_encoder * z
x = custom_normalize(x, axis=0)
print("X:", x)
