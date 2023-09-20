import time
import typing
import pathlib
import argparse
import matplotlib.pyplot as plt  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from . import printing
from text_parsing import InputData, ProbTensors
from hyperparameters import Hyperparameters
from model import Model


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
    font = {"weight": "normal", "size": 22}
    plt.figure(dpi=80, figsize=(25, 13))
    plt.plot(list(range(len(losses))), losses)
    plt.title('loss vs. step', fontdict=font)
    plt.xlabel('step', fontdict=font)
    plt.ylabel('loss', rotation=90, fontdict=font)
    plt.savefig('loss_vs_step.png')


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
):
    # Initialize the lists used for storing what we want to print to the terminal and google sheet
    losses = []  # Store all the losses for later printing
    minima_models_indices = []  # Store the indices of the models at the local minima so we can print just those

    # Initialize the optimizer and the loss function
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    total_loss = 0.0
    start = time.time()  # Keep track of time
    toc = start
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
            # Print the model parameters to a google sheet only if the boolean is set to true
            if not output_to_google_sheet:
                continue
            # Save the model parameters for later printing
            # Only output to the google sheet when we reach a local minimum
            if is_local_minimum(losses=losses, reached_local_minimum=reached_local_minimum):
                print("* LOSS went up *")

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
                    printing.send_to_google_sheet(
                        prompt_tensors=prompt_tensors,
                        preds=preds,
                        truths=truths,
                        token_prob_tensors=token_prob_tensors,
                        model=model,
                        attention_input=attention_input,
                        vocab=vocab,
                    )

                # Set the model back to training mode
                model.train()

            # If the loss goes back down for the first time, we have reached a local maximum
            elif len(losses) > 1 and losses[-1] < losses[-2] and not reached_local_maximum:
                reached_local_maximum = True
                reached_local_minimum = False

    # Plot the losses:
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
    return parser.parse_args()


def main():
    args = parse_args()
    data = InputData(args.training_text, args.inference_text)
    prob_tensors = ProbTensors(data=data, layer_width=args.layer_width)
    prompt_tensors = prob_tensors.make_inference_prompt_tensors(num_layers=args.num_layers)
    training_data = prob_tensors.format_training_data(num_layers=args.num_layers)
    hyperparams = Hyperparameters(
        layer_width=args.layer_width,
        num_data_points=args.num_data_points,
        num_layers=args.num_layers,
        weight_test=args.weight_test,
        perturbation_test=args.perturbation_test,
    )
    model = Model(hyperparams=hyperparams)

    # Train:
    if args.train:
        model = train_model(
            model=model,
            input_decision_activations=prob_tensors.decision_input,
            epochs=3,
            train_data=training_data,
            print_every=20,
            batch_size=2,
            lr=0.0001,
            vocab=data.vocab,
            prompt_tensors=prompt_tensors,
        )
    # Inference:
    elif prompt_tensors is not None:
        model.eval()  # set the model to inference mode
        printing.send_to_google_sheet(
            prompt_tensors,
            preds=None,
            truths=None,
            token_prob_tensors=None,
            model=model,
            input_decision_activations=prob_tensors.decision_input,
            vocab=data.vocab,
        )
    else:
        print("No inference prompt tensors found, so no inference will be run")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=20)
    tic = time.time()
    main()
    print("Total processing time:", time.time() - tic)
