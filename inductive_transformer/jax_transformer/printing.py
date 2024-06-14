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
                for position, position_weights in enumerate(layer_params[sublayer]["weights"]):
                    print(f"-- position {position}")
                    for token_num, token_weights in enumerate(position_weights):
                        if any(token_weights > 0.1):
                            print(f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000} -- {vocab[token_num]}")
                        else:
                            print(f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000}")
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
                for position, position_weights in enumerate(layer_params[sublayer]["weights"]):
                    print(f"-- position {position}")
                    for token_num, token_weights in enumerate(position_weights):
                        if any(token_weights > 0.1):
                            print(f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000} -- {vocab[token_num]}")
                        else:
                            print(f"{np.round(nn.relu(token_weights) * 1000).astype(int) / 1000}")
                    print()
            else:
                print(layer_params[sublayer]["weights"])
        print("")
