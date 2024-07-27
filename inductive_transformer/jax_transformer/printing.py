import numpy as np
from flax import linen as nn


def print_params(state, vocab):
    # Print trained weights.
    decoder_layers = ["decoders_0", "decoders_1"]
    encoder_layers = ["encoders_0", "encoders_1"]
    decoder_sublayers = [
        "decoder_attention_pi",
        "decoder_position_pi",
        "decoder_token_pi",
    ]
    encoder_sublayers = [
        "encoder_attention_pi",
        "encoder_position_pi",
        "encoder_token_pi",
    ]

    print("===================== Decoder Layers ======================")
    for layer in decoder_layers:
        print(layer)
        layer_params = state.params["params"][layer]
        for sublayer in decoder_sublayers:
            print(sublayer)
            if sublayer == "decoder_token_pi":
                for position, position_weights in enumerate(
                    layer_params[sublayer]["weights"]
                ):
                    print(f"-- position {position}")
                    for token_num, token_weights in enumerate(position_weights):
                        if any(token_weights > 0.1):
                            print(
                                f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000} -- {vocab[token_num]}"
                            )
                        else:
                            print(
                                f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000}"
                            )
                    print()
            else:
                print(layer_params[sublayer]["weights"])
        print("")

    print("===================== Encoder Layers ======================")
    for layer in encoder_layers:
        print(layer)
        layer_params = state.params["params"][layer]
        for sublayer in encoder_sublayers:
            print(sublayer)
            if sublayer == "encoder_token_pi":
                for position, position_weights in enumerate(
                    layer_params[sublayer]["weights"]
                ):
                    print(f"-- position {position}")
                    for token_num, token_weights in enumerate(position_weights):
                        if any(token_weights > 0.1):
                            print(
                                f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000} -- {vocab[token_num]}"
                            )
                        else:
                            print(
                                f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000}"
                            )
                    print()
            else:
                print(layer_params[sublayer]["weights"])
        print("")


def print_activations(
    n_examples, prompt_data, decoder_t, encoder_activations, decoder_activations
):
    print("===================== Inference Activations ======================")

    encoder_activation_keys = [
        "z",
        "u",
        "v",
        "y_categorical",
        "y_bernoulli",
        "rho_categorical",
        "x_categorical",
        "x_bernoulli",
        "z_prime",
    ]
    decoder_activation_keys = [
        "x_bernoulli",
        "y_bernoulli",
        "x_categorical",
        "y_categorical",
        "v",
        "rho_categorical",
        "t_categorical",
        "u",
        "z",
    ]

    for idx in range(n_examples)[::-1]:
        print("--------------------------")
        print(f"Inference example {idx}")

        print("input t")
        print(prompt_data[idx])
        print("output t")
        print(decoder_t[idx])
        print("")
        # Input from layer 0 to layer num_layers - 1
        # num_layers - 1 is the root
        # while layer 0 is the leaf
        for layer_idx, layer_activation in enumerate(encoder_activations):
            print("=" * 25)
            print(f"Layer {layer_idx} encoder")
            for key in encoder_activation_keys:  # type: ignore
                print(key)
                print(layer_activation[key][idx])
                print("")
        # Print the decoder activations in reverse order
        # since layer 0 is the leaf and num_layers - 1 is the root
        for layer_idx, layer_activation in enumerate(decoder_activations[::-1]):
            print("=" * 20)
            print(f"Layer {len(decoder_activations) - layer_idx - 1} decoder")
            for key in decoder_activation_keys:  # type: ignore
                print(key)
                print(layer_activation[key][idx])
                print("")
        print("--------------------------")
