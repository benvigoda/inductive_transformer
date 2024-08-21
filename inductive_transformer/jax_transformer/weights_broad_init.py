import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np  # type: ignore
from inductive_transformer.jax_transformer.helper_functions import EPSILON, get_num_layers
from synonyms import Synonyms  # type: ignore

strong = 1.0 - EPSILON  # Amplify the signal
weak = EPSILON  # Dampen the signal

mask_type = int


def set_position_pi_weights(
    layer,
    params,
    mask,
    prefix,
    layer_width,
    perturb_weights=False,
    lock_weights=True,
    noise_value=0.01,
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
            # Add a small amount of noise to the weights
            new_weights = new_weights + jax.random.normal(
                jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weights.shape
            ) * noise_value
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


def init_weights(
    params,
    vocab,
    lock_all_weights=False,
    perturb_weights=False,
    zero_out_right_weights=False,
    zero_out_left_weights=False,
    noise_value=0.01,
    catsanddogs=False,
):
    synonyms = Synonyms()
    if catsanddogs:
        synonyms.cats_and_dogs_overwrite()
    if zero_out_right_weights:
        synonyms.zero_right_words()
    if zero_out_left_weights:
        synonyms.zero_left_words()
    # Get shapes:
    num_positions, vocab_size, layer_width = params["params"]["encoders_0"][
        "encoder_token_pi"
    ]["weights"].shape
    assert vocab_size == len(vocab)
    """Update weights."""
    updated_params = params
    set_weights = jax.tree_util.tree_map(
        lambda x: jnp.ones_like(x, dtype=mask_type), params
    )

    num_layers = get_num_layers(params)

    for layer in range(num_layers):
        set_position_pi_weights(
            layer=layer,
            params=updated_params,
            mask=set_weights,
            perturb_weights=perturb_weights,
            lock_weights=lock_all_weights,
            prefix="decoder",
            layer_width=layer_width,
            noise_value=noise_value,
        )
        set_position_pi_weights(
            layer=layer,
            params=updated_params,
            mask=set_weights,
            perturb_weights=perturb_weights,
            lock_weights=lock_all_weights,
            prefix="encoder",
            layer_width=layer_width,
            noise_value=noise_value,
        )

    def set_token_weights(
        num_layer,
        layer_w,
        position,
        target_words,
        perturb_weights=False,
    ):
        """
        in the position where we want to listen for a particular word
        set a single word weight to strong and all the others to weak
        """
        # encoders_1 is layer=1
        new_weight_encoder = updated_params["params"][f"encoders_{num_layer}"]["encoder_token_pi"][
            "weights"
        ]
        new_weight_encoder = new_weight_encoder.at[position, :, layer_w].set(jnp.full(vocab_size, weak))

        for target_word in target_words:
            if target_word in vocab:
                vocab_idx = next(i for i, word in enumerate(vocab) if word.lower() == target_word)
                new_weight_encoder = new_weight_encoder.at[position, vocab_idx, layer_w].set(strong)
            else:
                # print(f"WARNING: Target word '{target_word}' not found in vocab")
                continue

        if perturb_weights:
            # Add a small amount of noise to the weights
            new_weight_encoder = new_weight_encoder + jax.random.normal(
                jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weight_encoder.shape
            ) * noise_value
        updated_params["params"][f"encoders_{num_layer}"]["encoder_token_pi"][
            "weights"
        ] = new_weight_encoder

        # decoder_1 is layer=1
        new_weight_decoder = updated_params["params"][f"decoders_{num_layer}"]["decoder_token_pi"][
            "weights"
        ]
        new_weight_decoder = new_weight_decoder.at[position, :, layer_w].set(jnp.full(vocab_size, weak))
        for target_word in target_words:
            if target_word in vocab:
                vocab_idx = next(i for i, word in enumerate(vocab) if word.lower() == target_word)
                new_weight_decoder = new_weight_decoder.at[position, vocab_idx, layer_w].set(strong)
            else:
                print(f"WARNING: Target word '{target_word}' not found in vocab")
                continue

        if perturb_weights:
            # Add a small amount of noise to the weights
            new_weight_decoder = new_weight_decoder + jax.random.normal(
                jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weight_decoder.shape
            ) * noise_value
        updated_params["params"][f"decoders_{num_layer}"]["decoder_token_pi"][
            "weights"
        ] = new_weight_decoder

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

            if perturb_weights:
                # Add a small amount of noise to the weights
                new_weight_encoder = new_weight_encoder + jax.random.normal(
                    jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weight_encoder.shape
                ) * noise_value
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
            if perturb_weights:
                # Add a small amount of noise to the weights
                new_weight_decoder = new_weight_decoder + jax.random.normal(
                    jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weight_decoder.shape
                ) * noise_value
            updated_params["params"][f"decoders_{num_layer}"]["decoder_token_pi"][
                "weights"
            ] = new_weight_decoder

    left_targets = synonyms.get_valid_left_ordered_words()
    right_targets = synonyms.get_valid_right_ordered_words()
    for pos in range(num_positions):
        for lw, target_words in zip(
            range(layer_width), [left_targets[pos], right_targets[pos]]
        ):  # [['big', 'large'], ['small']]
            num_lay = num_positions - pos - 1
            set_token_weights(num_lay, lw, pos, target_words, perturb_weights=perturb_weights)
            # print(updated_params["params"][f"decoders_{num_lay}"]["decoder_token_pi"]["weights"])
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

            if perturb_weights:
                # Add a small amount of noise to the weights
                new_weight = new_weight + jax.random.normal(
                    jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)), new_weight.shape
                ) * noise_value
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
