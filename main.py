import torch.nn.functional as F  # type: ignore
import torch  # type: ignore
import time
import copy
import matplotlib.pyplot as plt  # type: ignore
from input_text_parsing import InputData
from log_prob_tensors import LogProbTensors
import printing


class L2Loss:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __call__(self, pred, truth):
        loss = F.mse_loss(pred, truth, reduction="mean")
        return loss


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
    models_parameters = []  # Store the parameters of the models so we can print them to a google sheet

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
            # Print the model parameters to a google sheet
            if output_to_google_sheet:
                # Save the model parameters for later printing
                models_parameters.append(
                    {
                        "logical_encoder.decision_X1LSE.weights": copy.deepcopy([encoder_layer.decision_X1LSE.weights.detach().tolist() for encoder_layer in model.logical_encoder.layers]),
                        "logical_encoder.word_X1LSE.weights": copy.deepcopy([encoder_layer.word_X1LSE.weights.detach().tolist() for encoder_layer in model.logical_encoder.layers]),
                        "logical_decoder.concept_ALLSUM_decode.weights": copy.deepcopy([decoder_layer.concept_ALLSUM_decode.weights.detach().tolist() for decoder_layer in model.logical_decoder.layers]),
                        "logical_decoder.word_X1LSE_decode_layer.weights": copy.deepcopy([decoder_layer.word_X1LSE_decode_layer.weights.detach().tolist() for decoder_layer in model.logical_decoder.layers]),
                        "logical_decoder.pred": copy.deepcopy(preds.detach()),
                        "logical_decoder.truth": copy.deepcopy(truths.detach()),
                        "logical_decoder.input": copy.deepcopy(token_prob_tensors),
                        "to_decoder": model.to_decoder,
                    }
                )
                # Only output to the google sheet when we reach a local minimum
                if len(losses) > 1 and losses[-1] > losses[-2] and not reached_local_minimum:
                    print("* LOSS went up *")

                    reached_local_minimum = True
                    reached_local_maximum = False
                    minimum_index = i * epoch + i - 1
                    minima_models_indices.append(minimum_index)

                    model.eval()
                    z_decode = model.logical_decoder.z_decode_for_output

                    if prompt_tensors is not None:
                        # print(models_parameters[minimum_index]["logical_encoder.word_X1LSE.weights"])
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
                            word_log_prob_tensors=token_prob_tensors,
                            model=model,
                            models_parameters=models_parameters,
                            attention_input=attention_input,
                            vocab=vocab,
                            z_decode=z_decode,
                            minimum_index=minimum_index,
                        )


                    # safety measure if we failed to specify prompt tensor, we use first sentence in training data
                    else:
                        y = model(word_log_prob_tensors[0], attention_input[0])
                        res = printing.format_into_table(output=y, model=model, vocab=vocab, top_row=z_decode)
                        print('output to sheet')
                        printing.output_to_sheet(res, "Sheet1")

                    model.train()

                elif len(losses) > 1 and losses[-1] < losses[-2] and not reached_local_maximum:
                    reached_local_maximum = True
                    reached_local_minimum = False

                model.hyperparameters.print_word_weights = False
    return model
