from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F  # type: ignore
from jax_transformer.model import InductiveTransformer
from torch_transformer.hyperparameters import HyperParameters
from torch_transformer.model import Model
from torch_transformer.text_parsing import InputData, ProbTensors
from torch_transformer.main import get_model_weights


class L2Loss:
    def __call__(self, pred, truth):
        truth = truth.sum(dim=(0, -1))
        print("torch pred", pred.shape)
        print("torch truth", truth.shape)
        loss = F.mse_loss(pred, truth, reduction="mean")
        return loss


def jax_to_torch_tensor(jax_array):
    return torch.from_numpy(np.array(jax_array))


def torch_to_jax_tensor(torch_tensor):
    return jnp.array(torch_tensor.numpy())


def jax_loss_fn(model, params, z_in, t_in):
    z_out, t_out, encoder_activations, decoder_activations = model.apply(
        params, z_in, t_in
    )
    t_in = jnp.sum(t_in, axis=(0, -1))
    return jnp.mean(jnp.square(t_out - t_in))


def main():
    parent_dir = Path(__file__).resolve().parent

    # Initialize RNG state.
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    # seed = 1985637237
    key = jax.random.PRNGKey(seed)
    print(f"seed: {seed}\n")

    bernoulli_width = 2
    num_positions = 2
    vocab_size = 6
    layer_width = 2
    num_layers = 2

    key, subkey = jax.random.split(key)

    jax_model = InductiveTransformer(
        layer_width=layer_width,
        num_positions=num_positions,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )

    key, subkey_0, subkey_1, subkey_2 = jax.random.split(key, 4)
    z_in = jax.random.uniform(
        subkey_0, minval=0.0, maxval=1.0, shape=(bernoulli_width, layer_width)
    )
    t_in = jax.random.uniform(
        subkey_1,
        minval=0.0,
        maxval=1.0,
        shape=(num_layers, num_positions, vocab_size, layer_width),
    )
    jax_params = jax_model.init(subkey_2, z_in, t_in)

    # z_out, t_out, encoder_activations, decoder_activations = (
    #     jax_model.apply(jax_params, z_in, t_in)
    # )
    # print(z_out)

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

    if False:
        print("===================== Encoder Layers ======================")
        for layer in encoder_layers:
            print(layer)
            layer_jax_params = jax_params["params"][layer]
            for sublayer in encoder_sublayers:
                print(sublayer)
                print(layer_jax_params[sublayer]["weights"])
            print("")

        print("===================== Decoder Layers ======================")
        for layer in decoder_layers:
            print(layer)
            layer_jax_params = jax_params["params"][layer]
            for sublayer in decoder_sublayers:
                print(sublayer)
                print(layer_jax_params[sublayer]["weights"])
            print("")

    data = InputData(parent_dir / "training_text.txt", parent_dir / "inference_text.txt")
    prob_tensors = ProbTensors(data=data, layer_width=layer_width)
    torch_device = "cpu"
    prob_tensors.to(torch_device)
    hyperparams = HyperParameters(
        layer_width=layer_width,
        vocab_size=vocab_size,
        num_positions=prob_tensors.num_positions,
        num_layers=num_layers,
        weight_test=False,
        perturbation_test=False,
        init_perturb_weights=False,
    )

    # Overwrite the model weights that hyper-parameters set with the ones from jax:
    encoder_attention_pi_weights = torch.ones(num_layers, layer_width, layer_width)
    encoder_attention_pi_weights[0] = jax_to_torch_tensor(jax_params["params"]["encoders_0"]["encoder_attention_pi"]["weights"])
    encoder_attention_pi_weights[1] = jax_to_torch_tensor(jax_params["params"]["encoders_1"]["encoder_attention_pi"]["weights"])
    hyperparams.encoder_attention_pi_weights = encoder_attention_pi_weights
    encoder_position_pi_weights = torch.ones(num_layers, num_positions, layer_width)
    encoder_position_pi_weights[0] = jax_to_torch_tensor(jax_params["params"]["encoders_0"]["encoder_position_pi"]["weights"])
    encoder_position_pi_weights[1] = jax_to_torch_tensor(jax_params["params"]["encoders_1"]["encoder_position_pi"]["weights"])
    hyperparams.encoder_position_pi_weights = encoder_position_pi_weights
    encoder_token_pi_weights = torch.ones(num_layers, num_positions, vocab_size, layer_width)
    encoder_token_pi_weights[0] = jax_to_torch_tensor(jax_params["params"]["encoders_0"]["encoder_token_pi"]["weights"])
    encoder_token_pi_weights[1] = jax_to_torch_tensor(jax_params["params"]["encoders_1"]["encoder_token_pi"]["weights"])
    hyperparams.encoder_token_pi_weights = encoder_token_pi_weights
    decoder_attention_pi_weights = torch.ones(num_layers, layer_width, layer_width)
    decoder_attention_pi_weights[0] = jax_to_torch_tensor(jax_params["params"]["decoders_0"]["decoder_attention_pi"]["weights"])
    decoder_attention_pi_weights[1] = jax_to_torch_tensor(jax_params["params"]["decoders_1"]["decoder_attention_pi"]["weights"])
    hyperparams.decoder_attention_pi_weights = decoder_attention_pi_weights
    decoder_position_pi_weights = torch.ones(num_layers, num_positions, layer_width)
    decoder_position_pi_weights[0] = jax_to_torch_tensor(jax_params["params"]["decoders_0"]["decoder_position_pi"]["weights"])
    decoder_position_pi_weights[1] = jax_to_torch_tensor(jax_params["params"]["decoders_1"]["decoder_position_pi"]["weights"])
    hyperparams.decoder_position_pi_weights = decoder_position_pi_weights
    decoder_token_pi_weights = torch.ones(num_layers, num_positions, vocab_size, layer_width)
    decoder_token_pi_weights[0] = jax_to_torch_tensor(jax_params["params"]["decoders_0"]["decoder_token_pi"]["weights"])
    decoder_token_pi_weights[1] = jax_to_torch_tensor(jax_params["params"]["decoders_1"]["decoder_token_pi"]["weights"])
    hyperparams.decoder_token_pi_weights = decoder_token_pi_weights

    # Baseline sanity check...
    # print("encoder attention pi")
    # print(hyperparams.encoder_attention_pi_weights)
    # print("encoder position pi")
    # print(hyperparams.encoder_position_pi_weights)
    # print("encoder token pi")
    # print(hyperparams.encoder_token_pi_weights)
    # print("decoder attention pi")
    # print(hyperparams.decoder_attention_pi_weights)
    # print("decoder position pi")
    # print(hyperparams.decoder_position_pi_weights)
    # print("decoder token pi")
    # print(hyperparams.decoder_token_pi_weights)

    torch_model = Model(hyperparams=hyperparams)
    torch_model.eval()  # set the model to inference mode
    prompt_tensors = prob_tensors.make_inference_prompt_tensors(num_layers=num_layers)

    # lr=0.001
    # torch_optim = torch.optim.Adam(torch_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    torch_criterion = L2Loss()  # Simply use an L2 loss

    for prompt_tensor in prompt_tensors:
        attention_input = prob_tensors.attention_input
        torch_result = torch_model(attention_input, prompt_tensor)
        print("torch_model output", torch_result.shape)
        print(torch_result)
        torch_loss = torch_criterion(torch_result, prompt_tensor)
        print(f"torch loss: {torch_loss:.3e}")

        jax_attention = torch_to_jax_tensor(attention_input)
        jax_prompt = torch_to_jax_tensor(prompt_tensor)
        z_out, t_out, encoder_activations, decoder_activations = (
            jax_model.apply(jax_params, jax_attention, jax_prompt)
        )
        jax_loss = jax_loss_fn(jax_model, jax_params, jax_attention, jax_prompt)
        print("jax_model output")
        print("z_out", z_out.shape)
        print(z_out)
        print("t_in", jax_prompt.shape)
        print("t_out", t_out.shape)
        print(t_out)
        print(f"jax loss: {jax_loss:.3e}")

        print("")

    # print("encoder_layer_0.encoder_attention_pi.weights")
    # print(torch_model.encoder_layer_0.encoder_attention_pi.weights)
    # print("encoder_layer_1.encoder_attention_pi.weights")
    # print(torch_model.encoder_layer_1.encoder_attention_pi.weights)
    # print("encoder_layer_0.encoder_position_pi.weights")
    # print(torch_model.encoder_layer_0.encoder_position_pi.weights)
    # print("encoder_layer_1.encoder_position_pi.weights")
    # print(torch_model.encoder_layer_1.encoder_position_pi.weights)
    # print("encoder_layer_0.encoder_token_pi.weights")
    # print(torch_model.encoder_layer_0.encoder_token_pi.weights)
    # print("encoder_layer_1.encoder_token_pi.weights")
    # print(torch_model.encoder_layer_1.encoder_token_pi.weights)
    # print("decoder_layer_0.decoder_attention_pi.weights")
    # print(torch_model.decoder_layer_0.decoder_attention_pi.weights)
    # print("decoder_layer_1.decoder_attention_pi.weights")
    # print(torch_model.decoder_layer_1.decoder_attention_pi.weights)
    # print("decoder_layer_0.decoder_position_pi.weights")
    # print(torch_model.decoder_layer_0.decoder_position_pi.weights)
    # print("decoder_layer_1.decoder_position_pi.weights")
    # print(torch_model.decoder_layer_1.decoder_position_pi.weights)
    # print("decoder_layer_0.decoder_token_pi.weights")
    # print(torch_model.decoder_layer_0.decoder_token_pi.weights)
    # print("decoder_layer_1.decoder_token_pi.weights")
    # print(torch_model.decoder_layer_1.decoder_token_pi.weights)


if __name__ == "__main__":
    main()
