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

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np  # type: ignore
from jax_transformer.helper_functions import EPSILON, get_num_layers
from experimental_code.datasets.anavan import make_cat_dog_anavan, make_cat_dog_worm_bird_anavan  # type: ignore
from flax.core.frozen_dict import unfreeze, freeze  # Add import for unfreeze/freeze


strong = jnp.log(1.0 - EPSILON)  # Amplify the signal
weak = jnp.log(EPSILON)  # Dampen the signal
mask_type = jnp.float32  # Use float32 for consistency with parameters


def set_all_random(params):
    return jax.tree_util.tree_map(
        lambda x: jax.random.normal(
            jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)),
            x.shape
        ),
        params
    )


def set_attention_weights(p):
    """Set attention weights to 1 for straight-up connections and 0 for cross connections"""
    num_layers = get_num_layers(p)
    for layer in range(num_layers):
        for encoder_decoder in ["encoder", "decoder"]:
            layer_key = f"{encoder_decoder}s_{layer}"
            attention_pi = f"{encoder_decoder}_attention_pi"

            weights = p["params"][layer_key][attention_pi]["weights"]
            # Set all to weak first
            new_weights = jnp.full_like(weights, weak)
            # Set straight-up connections to strong
            new_weights = new_weights.at[0, 0].set(strong)  # left to left
            new_weights = new_weights.at[1, 1].set(strong)  # right to right
            p["params"][layer_key][attention_pi]["weights"] = new_weights
    return p


def set_position_weights(p):
    """Set position weights according to the layer pattern"""
    num_layers = get_num_layers(p)
    for layer in range(num_layers):
        for encoder_decoder in ["encoder", "decoder"]:
            layer_key = f"{encoder_decoder}s_{layer}"
            position_pi = f"{encoder_decoder}_position_pi"

            weights = p["params"][layer_key][position_pi]["weights"]
            new_weights = jnp.full_like(weights, weak)
            # Set the appropriate position to strong
            new_weights = new_weights.at[-layer - 1].set(
                jnp.full(weights.shape[-1], strong)
            )
            p["params"][layer_key][position_pi]["weights"] = new_weights

    return p


class SynonymList:
    def __init__(self, name, layer, position, layer_width_idx, token_list):
        self.name = name
        self.layer = layer
        self.position = position
        self.layer_width_idx = layer_width_idx
        self.token_list = token_list


class Synonyms:
    def __init__(self, vocab, catsanddogs=False):
        if catsanddogs:
            anavan = make_cat_dog_anavan()
        else:
            anavan = make_cat_dog_worm_bird_anavan()

        self.synonym_lists = [
            # Left side (layer_width_idx = 0)
            SynonymList("small",     5, 0, 0, anavan.get_synonyms_of_word("small", add_words={'wriggly'}, remove_words={'tiny'})),  # noqa: E241
            SynonymList("dogs",      4, 1, 0, anavan.get_synonyms_of_word("dogs")),  # noqa: E241
            SynonymList("often",     3, 2, 0, anavan.get_synonyms_of_word("often")),  # noqa: E241
            SynonymList("fear",      2, 3, 0, anavan.get_synonyms_of_word("fear")),  # noqa: E241
            SynonymList("large",     1, 4, 0, anavan.get_synonyms_of_word("large")),  # noqa: E241
            SynonymList("cats",      0, 5, 0, anavan.get_synonyms_of_word("cats")),  # noqa: E241

            # Right side (layer_width_idx = 1)
            SynonymList("wriggly",   5, 0, 1, anavan.get_synonyms_of_word("wriggly", add_words={'tiny'}, remove_words={'wriggly'})),  # noqa: E241
            SynonymList("worms",     4, 1, 1, anavan.get_synonyms_of_word("worms")),  # noqa: E241
            SynonymList("sometimes", 3, 2, 1, anavan.get_synonyms_of_word("sometimes")),  # noqa: E241
            SynonymList("chase",     2, 3, 1, anavan.get_synonyms_of_word("chase")),  # noqa: E241
            SynonymList("angry",     1, 4, 1, anavan.get_synonyms_of_word("angry")),  # noqa: E241
            SynonymList("birds",     0, 5, 1, anavan.get_synonyms_of_word("birds"))  # noqa: E241
        ]


def set_token_weights(params, synonyms, vocab):
    """Set token weights based on synonym lists"""
    for synonym_list in synonyms.synonym_lists:
        for encoder_decoder in ["encoder", "decoder"]:
            layer_key = f"{encoder_decoder}s_{synonym_list.layer}"
            token_pi = f"{encoder_decoder}_token_pi"

            if layer_key in params["params"]:
                weights = params["params"][layer_key][token_pi]["weights"]
                # Set weights for synonym tokens to strong
                for token in synonym_list.token_list:
                    if token in vocab:
                        vocab_idx = next(i for i, word in enumerate(vocab) if word.lower() == token)
                        weights = weights.at[synonym_list.position, vocab_idx, synonym_list.layer_width_idx].set(strong)
                params["params"][layer_key][token_pi]["weights"] = weights

    return params


'''class WeightTreeMapKeys:
    """ All possible index values
    encoder_decoder = ["encoder", "decoder"]
    factor = ["position", "token", "attention"]
    layer_width_idx = [0, 1] #left, right
    position_idx = [0, 1, 2, 3, 4, 5]
    token_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ... 53]
    weight_type = ["loose", "locked"]
    """
    def __init__(self, encoder_decoder, layer, factor, position_idx, token_idx, weight_type, noise_value):
        self.encoder_decoder = encoder_decoder
        self.layer = layer
        self.factor = factor
        self.position_idx = position_idx
        self.token_idx = token_idx
        self.noise_value = noise_value
        self.weight_type = weight_type

def add_perturbations(params, noise_value, perturb_indices=None):
    """Add noise to specified indices or all weights if indices not specified"""
    def add_noise(x):
        if perturb_indices is None:
            return x + jax.random.normal(
                jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)),
                x.shape
            ) * noise_value
        else:
            new_x = x.copy()
            for idx in perturb_indices:
                new_x = new_x.at[idx].add(noise_value)
            return new_x

    return jax.tree_util.tree_map(add_noise, params)
'''


def init_weights(
    params,
    vocab,
    noise_attention: float = 0.0,
    noise_position: float = 0.0,
    noise_token: float = 0.0,
    catsanddogs: bool = False,
    rng_key=None,
    lock_attention: bool = False,
    lock_position: bool = False,
    lock_token: bool = False,
):

    """Main function to set all weights and optionally add Gaussian noise per factor."""
    # Initialize all weights to weak
    p = unfreeze(params)

    p["params"] = jax.tree_util.tree_map(lambda x: jnp.full_like(x, weak), p["params"])

    # Set specific weight patterns
    p = set_attention_weights(p)
    p = set_position_weights(p)

    synonyms = Synonyms(vocab, catsanddogs=catsanddogs)
    p = set_token_weights(p, synonyms, vocab)

    # Add per-factor perturbations if specified
    if rng_key is None:
        rng_key = jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1))

    num_layers = get_num_layers(p)
    for factor_name, scale in (
        ("attention", noise_attention),
        ("position", noise_position),
        ("token", noise_token),
    ):
        if scale and scale > 0.0:
            for layer in range(num_layers):
                for encoder_decoder in ["encoder", "decoder"]:
                    layer_key = f"{encoder_decoder}s_{layer}"
                    factor_pi = f"{encoder_decoder}_{factor_name}_pi"
                    weights = p["params"][layer_key][factor_pi]["weights"]
                    rng_key, subkey = jax.random.split(rng_key)
                    noisy = weights + jax.random.normal(subkey, weights.shape) * scale
                    p["params"][layer_key][factor_pi]["weights"] = noisy

    weights_mask = jax.tree_util.tree_map(lambda x: jnp.ones_like(x, dtype=mask_type), params)

    # Apply per-factor locking by zeroing the corresponding mask leaves
    if lock_attention or lock_position or lock_token:
        num_layers = get_num_layers(p)
        for layer in range(num_layers):
            for encoder_decoder in ["encoder", "decoder"]:
                layer_key = f"{encoder_decoder}s_{layer}"
                if lock_attention:
                    factor_pi = f"{encoder_decoder}_attention_pi"
                    weights = p["params"][layer_key][factor_pi]["weights"]
                    weights_mask = unfreeze(weights_mask)
                    weights_mask["params"][layer_key][factor_pi]["weights"] = jnp.zeros_like(weights, dtype=mask_type)
                    weights_mask = freeze(weights_mask)
                if lock_position:
                    factor_pi = f"{encoder_decoder}_position_pi"
                    weights = p["params"][layer_key][factor_pi]["weights"]
                    weights_mask = unfreeze(weights_mask)
                    weights_mask["params"][layer_key][factor_pi]["weights"] = jnp.zeros_like(weights, dtype=mask_type)
                    weights_mask = freeze(weights_mask)
                if lock_token:
                    factor_pi = f"{encoder_decoder}_token_pi"
                    weights = p["params"][layer_key][factor_pi]["weights"]
                    weights_mask = unfreeze(weights_mask)
                    weights_mask["params"][layer_key][factor_pi]["weights"] = jnp.zeros_like(weights, dtype=mask_type)
                    weights_mask = freeze(weights_mask)

    return freeze(p), freeze(weights_mask)
