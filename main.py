import time
import math
import typing
import pathlib
import argparse
import torch  # type: ignore
from torch import nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR  # type: ignore
import printing
from text_parsing import InputData, ProbTensors
from hyperparameters import HyperParameters
from model import Model
from helper_functions import custom_normalize


def normalize_weights(weights):
    # return nn.functional.normalize(nn.ReLU()(weights), p=1, dim=0)
    return custom_normalize(nn.ReLU()(weights), dim=0)


class L2Loss:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __call__(self, pred, truth):
        loss = F.mse_loss(pred, truth, reduction="mean")
        return loss


def is_local_minimum(losses: typing.List[float], reached_local_minimum: bool = False) -> bool:
    if len(losses) > 1 and losses[-1] > losses[-2] and not reached_local_minimum:
        return True
    else:
        return False


def plot_convergence(losses: typing.List[float]):
    # Keeping the import here so we don't have to install matplotlib if we don't need it
    import matplotlib.pyplot as plt  # type: ignore
    font = {"weight": "normal", "size": 22}
    plt.figure(dpi=80, figsize=(25, 13))
    plt.plot(list(range(len(losses))), losses)
    plt.title('loss vs. step', fontdict=font)
    plt.xlabel('step', fontdict=font)
    plt.ylabel('loss', rotation=90, fontdict=font)
    plt.savefig('loss_vs_step.png')


def get_model_weights(model):
    encoder_attention_pi_weights = torch.stack([
        normalize_weights(model.encoder_layer_0.encoder_attention_pi.weights),
        normalize_weights(model.encoder_layer_1.encoder_attention_pi.weights),
    ], dim=0)
    encoder_token_pi_weights = torch.stack([
        normalize_weights(model.encoder_layer_0.encoder_token_pi.weights),
        normalize_weights(model.encoder_layer_1.encoder_token_pi.weights),
    ], dim=0)
    decoder_attention_pi_weights = torch.stack([
        normalize_weights(model.decoder_layer_0.decoder_attention_pi.weights),
        normalize_weights(model.decoder_layer_1.decoder_attention_pi.weights),
    ], dim=0)
    decoder_token_pi_weights = torch.stack([
        normalize_weights(model.decoder_layer_0.decoder_token_pi.weights),
        normalize_weights(model.decoder_layer_1.decoder_token_pi.weights),
    ], dim=0)
    model_weights = {
        'encoder_attention_pi_weights': encoder_attention_pi_weights,
        'encoder_token_pi_weights': encoder_token_pi_weights,
        'decoder_attention_pi_weights': decoder_attention_pi_weights,
        'decoder_token_pi_weights': decoder_token_pi_weights,
    }
    return model_weights


def train_model(
    model,
    attention_input,
    epochs,
    train_data,
    print_every=50,
    batch_size=2,
    lr=0.001,
    vocab=None,
    prompt_tensors=None,
    output_to_google_sheet=True,
    device=None,
    output_matplot=True,
):
    print("attention input", attention_input.device)
    # Initialize the lists used for storing what we want to print to the terminal and google sheet
    losses = []  # Store all the losses for later printing
    minima_models_indices = []  # Store the indices of the models at the local minima so we can print just those

    # Initialize the optimizer and the loss function
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler_plateau = ReduceLROnPlateau(optim, mode='min', factor=.1, patience=50, min_lr=5e-5, verbose=True)
    scheduler_cycle = CyclicLR(optim, base_lr=lr, max_lr=0.1, step_size_up=20, step_size_down=2, mode='triangular', cycle_momentum=False, verbose=True)
    total_loss = 0.0
    start = time.time()  # Keep track of time
    toc = start
    if device:
        model.to(device)
    model.train()  # Set the model to training mode

    criterion = L2Loss(optim)  # Simply use an L2 loss

    # Batch the training data
    # Training data is made of pairs of inputs and outputs. We split that into two lists
    n_batches = int(len(train_data) / batch_size)
    training_input = [t[0] for t in train_data]
    training_output = [t[1] for t in train_data]

    # Get the initial loss by running the model on the first batch
    token_prob_tensors = training_input[0: batch_size]
    truths = torch.stack(training_output[0: batch_size], 0)
    preds = torch.stack([model(attention_input, text_window) for text_window in token_prob_tensors], 0)
    initial_loss = criterion(preds, truths)
    print("Initial loss:", initial_loss)

    # Set the booleans to only print when local minimum is reached
    reached_local_minimum = False
    reached_local_maximum = False
    loss_avg = math.inf
    # Train the model for several epochs
    for epoch in range(epochs):
        for i in range(n_batches):
            # Feed each batch into the model
            token_prob_tensors = torch.stack(training_input[i * batch_size: (i + 1) * batch_size], 0)
            truths = torch.stack(training_output[i * batch_size: (i + 1) * batch_size], 0)
            preds = torch.stack([model(attention_input, text_window) for text_window in token_prob_tensors], 0)
            # Compute and save the loss for that batch
            loss = criterion(preds, truths)
            total_loss += loss
            losses.append(loss.detach())

            # Backpropagate the loss
            loss.backward()
            optim.step()
            # if 1500 < i % 2000 < 1600:
            #     scheduler_cycle.step()
            # else:
            #     scheduler_plateau.step(loss)

            # Print the loss every print_every batches
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                printing.print_to_terminal(
                    model=model,
                    iter=i,
                    epoch=epoch,
                    start=start,
                    loss_avg=loss_avg,
                    toc=toc,
                    print_every=print_every
                )
                toc = time.time()
                total_loss = 0
            # Save the model parameters for later printing
            # Only output to the google sheet when we reach a local minimum
            # Or at the very end of a batch
            if is_local_minimum(losses=losses, reached_local_minimum=reached_local_minimum) or i == n_batches - 1 or loss_avg < 1e-9:
                reached_local_minimum = True
                reached_local_maximum = False
                minimum_index = i * epoch + i - 1
                minima_models_indices.append(minimum_index)

                # Set the model to evaluation mode for inference
                model.eval()

                if prompt_tensors is not None:
                    printing.print_to_terminal(
                        model=model,
                        iter=i,
                        epoch=epoch,
                        start=start,
                        loss_avg=loss_avg,
                        toc=toc,
                        print_every=print_every,
                    )
                    if output_to_google_sheet:
                        printing.send_to_google_sheet(
                            prompt_tensors=prompt_tensors,
                            preds=preds,
                            truths=truths,
                            token_prob_tensors=token_prob_tensors,
                            model=model,
                            attention_input=attention_input,
                            vocab=vocab,
                        )

                # import pdb; pdb.set_trace()
                #############################
                # Use this to print the model parameters
                # encoder printing
                model.encoder_layer_0.encoder_universe.u
                model.encoder_layer_0.encoder_bernoulli_categorical.v
                model.encoder_layer_0.encoder_attention_pi.y
                model.encoder_layer_0.encoder_token_pi.x
                # model.encoder_layer_0.encoder_categorical_bernoulli.bernoulli
                model.encoder_layer_0.encoder_and.z
                model.encoder_layer_0.encoder_and.y
                model.encoder_layer_0.encoder_and.x

                model.encoder_layer_1.encoder_universe.u
                model.encoder_layer_1.encoder_bernoulli_categorical.v
                model.encoder_layer_1.encoder_attention_pi.y
                model.encoder_layer_1.encoder_token_pi.x
                # model.encoder_layer_1.encoder_categorical_bernoulli.bernoulli
                model.encoder_layer_1.encoder_and.z
                model.encoder_layer_1.encoder_and.y
                model.encoder_layer_1.encoder_and.x

                # decoder printing
                model.decoder_layer_1.decoder_and.y
                model.decoder_layer_1.decoder_and.x
                # model.decoder_layer_1.decoder_bernoulli_categorical.categorical
                model.decoder_layer_1.decoder_attention_pi.y
                model.decoder_layer_1.decoder_attention_pi.v
                model.decoder_layer_1.decoder_token_pi.t
                model.decoder_layer_1.decoder_categorical_bernoulli.u
                model.decoder_layer_1.decoder_universe.z

                model.decoder_layer_0.decoder_and.y
                model.decoder_layer_0.decoder_and.x
                # model.decoder_layer_0.decoder_bernoulli_categorical.categorical
                model.decoder_layer_0.decoder_attention_pi.y
                model.decoder_layer_0.decoder_attention_pi.y  # input
                model.decoder_layer_0.decoder_attention_pi.v  # output
                model.decoder_layer_0.decoder_token_pi.x  # input
                model.decoder_layer_0.decoder_token_pi.t  # output
                model.decoder_layer_0.decoder_categorical_bernoulli.u
                model.decoder_layer_0.decoder_universe.z
                #############################

                # Set the model back to training mode
                model.train()
                if loss_avg < 1e-9:
                    # Terminate training
                    return model

            # If the loss goes back down for the first time, we have reached a local maximum
            elif len(losses) > 1 and losses[-1] < losses[-2] and not reached_local_maximum:
                reached_local_maximum = True
                reached_local_minimum = False

    # Plot the losses:
    if output_matplot:
        plot_convergence(losses=losses)

    # Return the model
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Model arguments")
    parser.add_argument("training_text", type=pathlib.Path)  # A text file of sentences to train on
    parser.add_argument("inference_text", type=pathlib.Path)  # A text file of sentences to run inference on
    parser.add_argument(
        "--train", help="Whether to train model or not", action="store_true"
    )
    parser.add_argument(
        "--weight_test", help="Whether to use test weights", action="store_true",
    )
    parser.add_argument(
        "--perturbation_test", help="Whether to use only some weights for perturbation", action="store_true",
    )
    parser.add_argument("--layer_width", type=int, default=4)
    parser.add_argument("--num_data_points", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_train", type=int, default=1)  # Number of times to train
    parser.add_argument("--silence_google_sheet", help="Whether to output to google sheet", action="store_true")
    parser.add_argument("--use_gpu", help="Whether to run on a GPU if available", action="store_true")
    parser.add_argument("--silence_matplot", help="Whether to output matplotlib plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cpu"
    if args.use_gpu and torch.cuda.is_available():
        device = "cuda:0"
    data = InputData(args.training_text, args.inference_text)
    prob_tensors = ProbTensors(data=data, layer_width=args.layer_width)
    prob_tensors.to(device)
    training_data = prob_tensors.format_training_data(num_layers=args.num_layers, device=device)
    inference_match_training = True  # Toggle to match training data or not
    if inference_match_training:
        prompt_tensors = [input_training for input_training, _ in training_data]
    else:
        prompt_tensors = prob_tensors.make_inference_prompt_tensors(num_layers=args.num_layers)
    training_data = training_data * math.ceil(args.num_data_points / len(training_data))  # Duplicate to have num_data_points
    hyperparams = HyperParameters(
        layer_width=args.layer_width,
        vocab_size=data.vocab_size,
        num_positions=prob_tensors.num_positions,
        num_layers=args.num_layers,
        weight_test=args.weight_test,
        perturbation_test=args.perturbation_test,
    )
    model = Model(hyperparams=hyperparams)

    # Train:
    if args.train:
        encoder_attention_pi_weights = []
        encoder_token_pi_weights = []
        decoder_attention_pi_weights = []
        decoder_token_pi_weights = []
        for i in range(args.num_train):
            print(f"Training run {i + 1} of {args.num_train}")
            model = Model(hyperparams=hyperparams)
            model = train_model(
                model=model,
                attention_input=prob_tensors.attention_input,
                epochs=1,
                train_data=training_data,
                print_every=20,
                batch_size=len(prob_tensors.windows),  # Batch all the different sentences together
                lr=1e-3,
                vocab=data.vocab,
                prompt_tensors=prompt_tensors,
                output_to_google_sheet=not args.silence_google_sheet,
                device=device,
                output_matplot=not args.silence_matplot,
            )
            model_weights = get_model_weights(model=model)
            encoder_attention_pi_weights.append(model_weights['encoder_attention_pi_weights'])
            encoder_token_pi_weights.append(model_weights['encoder_token_pi_weights'])
            decoder_attention_pi_weights.append(model_weights['decoder_attention_pi_weights'])
            decoder_token_pi_weights.append(model_weights['decoder_token_pi_weights'])
        # Stack-up the lists to make tensors we can run stats on
        encoder_attention_pi_weights = torch.stack(encoder_attention_pi_weights, dim=0)
        encoder_token_pi_weights = torch.stack(encoder_token_pi_weights, dim=0)
        decoder_attention_pi_weights = torch.stack(decoder_attention_pi_weights, dim=0)
        decoder_token_pi_weights = torch.stack(decoder_token_pi_weights, dim=0)
        # Print the mean and standard deviation of the weights
        print("encoder_attention_pi_weights:")
        print(encoder_attention_pi_weights.mean(dim=0))
        print("+/-")
        print(encoder_attention_pi_weights.std(dim=0))
        print("encoder_token_pi_weights:")
        print(encoder_token_pi_weights.mean(dim=0))
        print("+/-")
        print(encoder_token_pi_weights.std(dim=0))
        print("decoder_attention_pi_weights:")
        print(decoder_attention_pi_weights.mean(dim=0))
        print("+/-")
        print(decoder_attention_pi_weights.std(dim=0))
        print("decoder_token_pi_weights:")
        print(decoder_token_pi_weights.mean(dim=0))
        print("+/-")
        print(decoder_token_pi_weights.std(dim=0))
        # import pdb; pdb.set_trace()

    # Inference:
    elif prompt_tensors is not None:
        model.eval()  # set the model to inference mode
        # printing.send_to_google_sheet(
        #     prompt_tensors,
        #     preds=None,
        #     truths=None,
        #     token_prob_tensors=None,
        #     model=model,
        #     attention_input=prob_tensors.attention_input,
        #     vocab=data.vocab,
        # )
    else:
        print("No inference prompt tensors found, so no inference will be run")


if __name__ == "__main__":
    # As of November 2023, using float64 breaks optimization on the GPU.
    # https://discuss.pytorch.org/t/tensors-of-the-same-index-must-be-on-the-same-device-and-the-same-dtype-except-step-tensors-that-can-be-cpu-and-float32-notwithstanding/190335
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=20)
    tic = time.time()
    main()
    print("Total processing time:", time.time() - tic)
