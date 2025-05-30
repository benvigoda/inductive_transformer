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

import jax
import jax.numpy as jnp


def sample(key: jax.Array, decoder_t: jax.Array, temperature: float = 1.0):
    assert decoder_t.ndim == 2
    num_positions, vocab_size = decoder_t.shape

    # # The categorical distribution over tokens (at each position) should sum to one.
    # sums = decoder_t.sum(axis=1)
    # if not jnp.allclose(sums, 1.0):
    #     print("WARNING: categorical distributions do not all sum to one")
    #     print(sums)

    # For now we just generate a single sample.
    # TODO: Allow taking multiple (independent) samples.
    log_probs = decoder_t / temperature
    samples = jax.random.categorical(key, log_probs, axis=-1)
    return samples
