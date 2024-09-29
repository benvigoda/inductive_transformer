import numpy as np  # type: ignore
from flax import linen as nn  # type: ignore
from helper_functions import get_num_layers  # type: ignore


def print_params(state, vocab, silence_print=False):
    text = ""
    np.set_printoptions(threshold=np.inf)
    num_layers = get_num_layers(state.params)
    # Print trained weights.
    text += "===================== Model WEIGHTS ======================\n"
    decoder_layers = [f"decoders_{i}" for i in range(num_layers)]
    encoder_layers = [f"encoders_{i}" for i in range(num_layers)]
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

    text += "===================== Decoder Layers ======================\n"
    for layer in decoder_layers:
        text += f"{layer}\n"
        layer_params = state.params["params"][layer]
        for sublayer in decoder_sublayers:
            text += f"{sublayer}\n"
            if sublayer == "decoder_token_pi":
                for position, position_weights in enumerate(
                    layer_params[sublayer]["weights"]
                ):
                    text += f"-- position {position}\n"
                    for token_num, token_weights in enumerate(position_weights):
                        if any(token_weights > 0.1):
                            # text += f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000} -- {vocab[token_num]}\n"
                            text += f"{token_weights} -- {vocab[token_num]}\n"
                        else:
                            # text += f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000}\n"
                            text += f"{token_weights}\n"
                    text += "\n"
            else:
                text += f"{layer_params[sublayer]['weights']}\n"
        text += "\n"
    text += "===================== Encoder Layers ======================\n"
    for layer in encoder_layers:
        text += f"{layer}\n"
        layer_params = state.params["params"][layer]
        for sublayer in encoder_sublayers:
            text += f"{sublayer}\n"
            if sublayer == "encoder_token_pi":
                for position, position_weights in enumerate(
                    layer_params[sublayer]["weights"]
                ):
                    text += f"-- position {position}\n"
                    for token_num, token_weights in enumerate(position_weights):
                        if any(token_weights > 0.1):
                            # text += f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000} -- {vocab[token_num]}\n"
                            text += f"{token_weights} -- {vocab[token_num]}\n"
                        else:
                            # text += f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000}\n"
                            text += f"{token_weights}\n"
                    text += "\n"
            else:
                text += f"{layer_params[sublayer]['weights']}\n"
        text += "\n"
    if not silence_print:
        print(text)
    return text


def print_activations(
    n_examples, prompt_data, decoder_t, encoder_activations, decoder_activations, silence_print=False
):
    text = ""
    np.set_printoptions(threshold=np.inf)
    text += "===================== Inference Activations ======================\n"

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
        text += "--------------------------\n"
        text += f"Inference example {idx}\n"

        text += "input t\n"
        text += f"{prompt_data[idx]}\n"
        text += "output t\n"
        text += f"{decoder_t[idx]}\n"
        text += "\n"
        # Input from layer 0 to layer num_layers - 1
        # num_layers - 1 is the root
        # while layer 0 is the leaf
        for layer_idx, layer_activation in enumerate(encoder_activations):
            text += "=" * 25 + "\n"
            text += f"Layer {layer_idx} encoder\n"
            for key in encoder_activation_keys:  # type: ignore
                text += f"{key}\n"
                text += f"{layer_activation[key][idx]}\n"
                text += "\n"
        # Print the decoder activations in reverse order
        # since layer 0 is the leaf and num_layers - 1 is the root
        for layer_idx, layer_activation in enumerate(decoder_activations[::-1]):
            text += "=" * 20 + "\n"
            text += f"Layer {len(decoder_activations) - layer_idx - 1} decoder\n"
            for key in decoder_activation_keys:  # type: ignore
                text += f"{key}\n"
                text += f"{layer_activation[key][idx]}\n"
                text += "\n"
        text += "--------------------------\n"
    if not silence_print:
        print(text)
    return text
