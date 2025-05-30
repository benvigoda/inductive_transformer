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
from jax_transformer.helper_functions import EPSILON, get_num_layers, PROBABLE, IMPROBABLE
from inductive_transformer.datasets.anavan import make_cat_dog_anavan, make_cat_dog_worm_bird_anavan  # type: ignore

strong = jnp.log(1.0 - EPSILON)  # Amplify the signal
weak = jnp.log(EPSILON)  #= IMPROBABLE# Dampen the signal
mask_type = jnp.float32



def set_position_pi_weights(
    layer,
    params,
    mask,
    prefix,
    layer_width,
    perturb_weights=False,
    lock_weights=False,
    noise_value=0.01,
    surgical_perturb=False,
):
    layer_key = f"{prefix}s_{layer}"
    position_pi = f"{prefix}_position_pi"
    if layer_key in params["params"]:
        # Get the shape of the original weights
        old_weights = params["params"][layer_key][position_pi]["weights"]
        # Set the weights to all "weak" values
        new_weights = jnp.full(old_weights.shape, weak)  # (num_positions, layer_width)

        new_weights = new_weights.at[-layer - 1].set(
            jnp.full(layer_width, strong)
        )  # Match num_layer to the opposite position in the weights

        if perturb_weights:
            if not surgical_perturb:
                # Add a small amount of noise to the weights
                new_weights = new_weights + jax.random.normal(
                    jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weights.shape
                ) * noise_value
            else:
                new_weights = new_weights.at[[0][0]].set(new_weights[0][0] + noise_value)

        params["params"][layer_key][position_pi]["weights"] = new_weights
        # Note: We have constrained the model such that num_positions needs to be equal
        # to num_layers for this setup to work right now.
        # In the future we'll have to remove that constraint

        if lock_weights:
            mask["params"][layer_key][position_pi]["weights"] = jnp.zeros_like(
                old_weights, dtype=mask_type
            )
    else:
        raise ValueError(f"Layer {layer_key} not found in params.")



def set_encoder_decoder_token_weights(
    updated_params,
    vocab,
    num_layer,
    layer_w,
    position,
    target_words,
    prefix
):
    """
    in the position where we want to listen for a particular word
    set a single word weight to strong and all the others to weak
    """
    vocab_size = len(vocab)
    prefix_s = prefix + "s"
    # encoders_1 is layer=1
    new_weight_encoder_decoder = updated_params["params"][f"{prefix_s}_{num_layer}"][f"{prefix}_token_pi"][
        "weights"
    ]
    new_weight_encoder_decoder = new_weight_encoder_decoder.at[position, :, layer_w].set(jnp.full(vocab_size, weak))

    for target_word in target_words:
        if target_word in vocab:
            vocab_idx = next(i for i, word in enumerate(vocab) if word.lower() == target_word)
            new_weight_encoder_decoder = new_weight_encoder_decoder.at[position, vocab_idx, layer_w].set(strong)
        else:
            # print(f"WARNING: Target word '{target_word}' not found in vocab")
            continue
    return new_weight_encoder_decoder


def perturb_weights_func(
    updated_params,
    num_layer,
    perturb_weights,
    perturb_token,
    surgical_perturb,
    noise_value,
    weights,
    prefix,
):
    prefix_s = prefix + "s"
    if perturb_weights or perturb_token:
        if not surgical_perturb:
            # Add a small amount of noise to the weights
            weights = weights + jax.random.normal(
                jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), weights.shape
            ) * (noise_value if perturb_weights else perturb_token)
    return weights


def init_weights(
    params,
    vocab,
    lock_all_weights=False,
    perturb_weights=False,
    perturb_position=None,
    perturb_token=None,
    perturb_attention=None,
    surgical_perturb=False,
    noise_value=0.01,
    catsanddogs=False,
):
    if catsanddogs:
        synonyms = make_cat_dog_anavan()
    else:
        synonyms = make_cat_dog_worm_bird_anavan()
    # Get shapes:
    num_positions, vocab_size, layer_width = params["params"]["encoders_0"][
        "encoder_token_pi"]["weights"].shape
    assert vocab_size == len(vocab)
    

    """Shape the weights"""
    updated_params = params
    set_weights = jax.tree_util.tree_map(
        lambda x: jnp.ones_like(x, dtype=mask_type), params
    )
    num_layers = get_num_layers(params)


    def set_token_weights(
        num_layer,
        layer_w,
        position,
        target_words,
        perturb_weights=False,
        surgical_perturb=False,
        noise_value=0.01,
    ):
        """
        in the position where we want to listen for a particular word
        set a single word weight to strong and all the others to weak
        """
        prefix = 'encoder'
        new_weight_encoder = set_encoder_decoder_token_weights(
            updated_params=updated_params, vocab=vocab, num_layer=num_layer,
            layer_w=layer_w, position=position, target_words=target_words, prefix=prefix)
        new_weight_encoder = perturb_weights_func(updated_params=updated_params, num_layer=num_layer, surgical_perturb=surgical_perturb, noise_value=noise_value, prefix=prefix)
        
        updated_params["params"][f"{prefix_s}_{num_layer}"][f"{prefix}_token_pi"][
        "weights"] = weights

        prefix = 'decoder'
        new_weight_decoder = set_encoder_decoder_token_weights(
            updated_params=updated_params, vocab=vocab, num_layer=num_layer,
            layer_w=layer_w, position=position, target_words=target_words, prefix=prefix)
        new_weight_decoder = perturb_weights_func(updated_params=updated_params, num_layer=num_layer, surgical_perturb=surgical_perturb, noise_value=noise_value, prefix=prefix)




        """
        in the position where we want to listen for NO word
        set all weights to weak
        """
        for other_position in range(num_positions):
            if other_position == position:
                continue
            # encoders_1 is layer=1
            new_weight_encoder = updated_params["params"][f"encoders_{num_layer}"]["encoder_token_pi"][
                "weights"
            ]
            new_weight_encoder = new_weight_encoder.at[other_position, :, layer_w].set(
                jnp.full(vocab_size, weak)
            )

            if perturb_weights or perturb_token:
                if not surgical_perturb:
                    # Add a small amount of noise to the weights
                    new_weight_encoder = new_weight_encoder + jax.random.normal(
                        jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weight_encoder.shape
                    ) * (noise_value if perturb_weights else perturb_token)
            updated_params["params"][f"encoders_{num_layer}"]["encoder_token_pi"][
                "weights"
            ] = new_weight_encoder

            # decoder_1 is layer=1
            new_weight_decoder = updated_params["params"][f"decoders_{num_layer}"]["decoder_token_pi"][
                "weights"
            ]
            new_weight_decoder = new_weight_decoder.at[other_position, :, layer_w].set(
                jnp.full(vocab_size, weak)
            )
            if perturb_weights or perturb_token:
                if not surgical_perturb:
                    # Add a small amount of noise to the weights
                    new_weight_decoder = new_weight_decoder + jax.random.normal(
                        jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weight_decoder.shape
                    ) * (noise_value if perturb_weights else perturb_token)
            updated_params["params"][f"decoders_{num_layer}"]["decoder_token_pi"][
                "weights"
            ] = new_weight_decoder


    for layer in range(num_layers):
        set_position_pi_weights(
            layer=layer,
            params=updated_params,
            mask=set_weights,
            perturb_weights=perturb_weights or bool(perturb_position),
            lock_weights=False,
            prefix="decoder",
            layer_width=layer_width,
            noise_value=noise_value if perturb_weights else perturb_position,
            surgical_perturb=surgical_perturb,
        )
        set_position_pi_weights(
            layer=layer,
            params=updated_params,
            mask=set_weights,
            perturb_weights=perturb_weights or bool(perturb_position),
            lock_weights=False,
            prefix="encoder",
            layer_width=layer_width,
            noise_value=noise_value if perturb_weights else perturb_position,
            surgical_perturb=surgical_perturb,
        )

    left_targets = synonyms.get_valid_left_ordered_words()
    right_targets = synonyms.get_valid_right_ordered_words()
    for pos in range(num_positions):
        for lw, target_words in zip(
            range(layer_width), [left_targets[pos], right_targets[pos]]
        ):  # [['big', 'large'], ['small']]
            num_lay = num_positions - pos - 1
            set_token_weights(
                num_layer=num_lay,
                layer_w=lw,
                position=pos,
                target_words=target_words,
                perturb_weights=perturb_weights or bool(perturb_token),
                noise_value=noise_value if perturb_weights else perturb_token,
                surgical_perturb=surgical_perturb,
            )

            # Fix set_weights so the gradient does not update the weights
            if lock_all_weights:
                set_weights["params"][f"encoders_{num_lay}"]["encoder_token_pi"][
                    "weights"
                ] = jnp.zeros_like(
                    updated_params["params"][f"encoders_{num_lay}"]["encoder_token_pi"][
                        "weights"
                    ],
                    dtype=mask_type,
                )
                set_weights["params"][f"decoders_{num_lay}"]["decoder_token_pi"][
                    "weights"
                ] = jnp.zeros_like(
                    updated_params["params"][f"decoders_{num_lay}"]["decoder_token_pi"][
                        "weights"
                    ],
                    dtype=mask_type,
                )

    """Set attention weights."""
    set_flat_weights = False
    # we want no cross connections
    # for the connection between layer=1 and layer=0,
    # in lw=0, attention weight connecting to lw=0 should be strong and connecting to lw=1 weak
    # in lw=1, attention weight connecting to lw=0 should be weak and connecting to lw=1 strong
    # for the connection from layer=0 to the inputs/outputs, we can have all the weights be uniform
    for encoder_decoder in ["encoder", "decoder"]:
        encoders_decoders = encoder_decoder + "s"
        for layer in range(num_layers):
            new_weight = updated_params["params"][f"{encoders_decoders}_{layer}"][
                f"{encoder_decoder}_attention_pi"
            ]["weights"]
            if set_flat_weights:
                new_weight = new_weight.at[:, :].set(
                    jnp.full((layer_width, layer_width), strong / 2)
                )
            else:
                new_weight = new_weight.at[:, :].set(
                    jnp.full((layer_width, layer_width), weak)
                )
                new_weight = new_weight.at[1, 1].set(strong)
                new_weight = new_weight.at[0, 0].set(strong)

            # if perturb_weights or perturb_attention:
            #     if not surgical_perturb:
            #         # Add a small amount of noise to the weights
            #         new_weights = new_weights + jax.random.normal(
            #             jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weights.shape
            #         ) * noise_value
            #     else:
            #         new_weights = new_weights.at[[0][0]].set(new_weights[0][0] - jax.random.normal(jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1))) * noise_value)

            if perturb_weights or perturb_attention:
                if not surgical_perturb:
                    # Add a small amount of noise to the weights
                    new_weight = new_weight + jax.random.normal(
                        jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weight.shape
                    ) * (noise_value if perturb_weights else perturb_attention)
            updated_params["params"][f"{encoders_decoders}_{layer}"][
                f"{encoder_decoder}_attention_pi"
            ]["weights"] = new_weight

            # Fix set_weights so the gradient does not update the weights
            if lock_all_weights:
                set_weights["params"][f"{encoders_decoders}_{layer}"][
                    f"{encoder_decoder}_attention_pi"
                ]["weights"] = jnp.zeros_like(
                    updated_params["params"][f"{encoders_decoders}_{layer}"][
                        f"{encoder_decoder}_attention_pi"
                    ]["weights"],
                    dtype=mask_type,
                )

    return updated_params, set_weights
