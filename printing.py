import torch  # type: ignore
import pathlib
import os
import time
from typing import List
from torch import nn  # type: ignore
from helper_functions import custom_normalize


def format_prob_vocab(prob_outputs, vocab):
    """
    Takes in a prob_outputs tensor and a list of vocab tokens
    It assumes the two are sorted in corresponding order
    (meaning) that the first entry in the prob_outputs tensors
    does correspond to the first token in the vocab list

    Return a string corresponding
    to the list of formatted tuples linking each prob to it's corresponding
    token, sorted from highest prob to lowest
    """
    # Pair each prob with it's corresponding token string in the vocab
    prob_token_pairs = [(round(x.item() * 100) / 100, v) for x, v in zip(prob_outputs, vocab)]
    # Sort from highest probability to lowest, so we have the most likely token on top
    prob_token_pairs.sort(key=lambda x: x[0], reverse=True)
    prob_token_pairs_str = "\n".join([str(pw) for pw in prob_token_pairs])
    return prob_token_pairs_str


def format_attention_tensor(t):
    return '[' + ', '.join(['{:.2f}'.format(value) for value in t]) + ']'


def print_to_terminal(model, iter, epoch, start, loss_avg, toc, print_every):
    print(
        "time = %dm, epoch %d, iter = %d, loss = %.10f,\
        %ds per %d iters"
        % (
            (time.time() - start) // 60,
            epoch + 1,
            iter + 1,
            loss_avg,
            time.time() - toc,
            print_every,
        )
    )


def normalize_weights(weights, dim=1):
    # return nn.functional.normalize(nn.ReLU()(weights), p=1, dim=0)
    return custom_normalize(nn.ReLU()(weights), dim=dim)


def send_to_google_sheet(prompt_tensors, preds, truths, token_prob_tensors, model, attention_input, vocab):
    # token_prob_tensors are the training inputs
    prompt_preds = []  # Store the predictions from each prompt_tensor
    encoder_attention_pi_weights = torch.stack([
        normalize_weights(model.encoder_layer_0.encoder_attention_pi.weights),
        normalize_weights(model.encoder_layer_1.encoder_attention_pi.weights),
    ], dim=0)
    encoder_token_pi_weights = torch.stack([
        normalize_weights(model.encoder_layer_0.encoder_token_pi.weights),
        normalize_weights(model.encoder_layer_1.encoder_token_pi.weights),
    ], dim=0)
    decoder_attention_pi_weights = torch.stack([
        normalize_weights(model.decoder_layer_0.decoder_attention_pi.weights, dim=1),
        normalize_weights(model.decoder_layer_1.decoder_attention_pi.weights, dim=1),
    ], dim=0)
    decoder_token_pi_weights = torch.stack([
        normalize_weights(model.decoder_layer_0.decoder_token_pi.weights, dim=1),
        normalize_weights(model.decoder_layer_1.decoder_token_pi.weights, dim=1),
    ], dim=0)
    for sheet_number, prompt_tensor in enumerate(prompt_tensors):
        # write activations
        y = model(attention_input, prompt_tensor)
        if sheet_number == 0:
            # write decoder weights
            res_decoder_weights = format_into_table(
                output=decoder_token_pi_weights,
                model=model,
                vocab=vocab,
                top_row=decoder_attention_pi_weights,
                top_row_label="decoder attention pi weights",
                decoder=True,
            )
            print("decoder weights")
            output_to_sheet(res_decoder_weights, "decoder_weights")

            # write encoder weights
            res_encoder_weights = format_into_table(
                output=encoder_token_pi_weights,
                model=model,
                vocab=vocab,
                top_row=encoder_attention_pi_weights,
                top_row_label="encoder attention pi weights",
                decoder=False,
            )
            print("encoder weights")
            output_to_sheet(res_encoder_weights, "encoder_weights")

        prompt_preds.append(y)
    ############################
    # TODO: We currently do not track the attention_preds and attention_truths
    # Should we?
    # So, for now, we just keep them empty
    attention_preds = [torch.empty(attention_input.shape) for _ in range(len(prompt_tensors))]
    attention_truths = [torch.empty(attention_input.shape) for _ in range(len(prompt_tensors))]
    ############################
    # With test data sentences running through inference
    # Print the input, pred, and truth (same as input) for each activation:
    res_pred_truth_input = format_into_pred_truth_table(
        model=model,
        vocab=vocab,
        preds=prompt_preds,
        truths=truths,
        inputs=prompt_tensors,
        attention_preds=attention_preds,
        attention_truths=attention_truths,
        attention_inputs=[attention_input for _ in range(len(prompt_tensors))],
        title="test prediction versus truth",
    )
    print("inference_input_pred")
    output_to_sheet(res_pred_truth_input, "inference_input_pred")

    if preds is not None and truths is not None and token_prob_tensors is not None:
        res_pred_truth_input = format_into_pred_truth_table(
            model=model,
            vocab=vocab,
            preds=preds,
            truths=truths,
            inputs=token_prob_tensors,
            attention_preds=attention_preds,
            attention_truths=attention_truths,
            attention_inputs=[attention_input for _ in range(len(prompt_tensors))],
            title="training prediction versus truth"
        )
        print("training_input_pred_truth")
        output_to_sheet(res_pred_truth_input, "training_input_pred_truth")


def format_into_pred_truth_table(model, vocab, preds, truths, inputs, attention_preds, attention_truths, attention_inputs, title=""):
    # preds, truths, inputs are all size (num_sentences, num_layers, vocab_size, layer_width)
    # attention_preds, attention_truths, attention_inputs are all size (num_sentences, layer_width, layer_width)
    # The num_layers index is collapsed as we are only training on the 0th output layer

    # Initialize the table to None
    table = [[None for _ in range(1 + len(preds) * (model.hyperparams.layer_width * 2 + 1))] for _ in range(model.hyperparams.num_layers * 4 + 1)]
    # Title in the top left corner
    table[0][0] = title

    # Label each sentence
    for k in range(len(preds)):
        # sentence 0 is in column 1
        # sentence 1 is in column 1 + layer_width + 1
        # sentence 2 is in column 1 + layer_width + 1 + layer_width + 1
        # ...
        table[0][1 + k * (model.hyperparams.layer_width * 2 + 1)] = f"sentence {k}"
    # Fill-in the token-lists
    for n in range(model.hyperparams.num_layers):
        table[1 + n * 4][0] = f"pred layer {n}"
        table[1 + n * 4 + 1][0] = f"truth layer {n}"
        table[1 + n * 4 + 2][0] = f"input layer {n}"
        # Leave the 4th line blank
        for lw in range(model.hyperparams.layer_width):
            for k in range(len(preds)):
                # row = n * 4
                # for sentence 0: columns are 1 + lw
                # for sentence 1: columns are 1 + layer_width + 1 + lw
                # for sentence 2: columns are 1 + layer_width + 1 + num_layers + 1 + lw
                # ...
                token_row = (1 + model.hyperparams.layer_width * 2) * k + 1 + lw
                attention_row = token_row + model.hyperparams.layer_width
                table[1 + n * 4][token_row] = format_prob_vocab(preds[k][n][0, :, lw], vocab)  # FIXME: print all positions not just the 0th one
                table[1 + n * 4 + 1][token_row] = format_prob_vocab(truths[k][n][0, :, lw], vocab)  # FIXME: print all positions not just the 0th one
                table[1 + n * 4 + 2][token_row] = format_prob_vocab(inputs[k][n][0, :, lw], vocab)  # FIXME: print all positions not just the 0th one
                # For the attention output we are only interested in the 0th layer
                # We are not training on any other layer
                # So the `n` index is dropped
                if n == 0:
                    table[1 + n * 4][attention_row] = format_attention_tensor(attention_preds[k][:, lw])
                    table[1 + n * 4 + 1][attention_row] = format_attention_tensor(attention_truths[k][:, lw])
                    table[1 + n * 4 + 2][attention_row] = format_attention_tensor(attention_inputs[k][:, lw])
    return table


def format_into_table(output, model, vocab, top_row, top_row_label: str, decoder: bool = False):
    # Takes in the inference output of a model and formats it into a table

    ############################
    # Legacy code to support the lack of position sensitivity
    if output.shape == (model.hyperparams.num_layers, model.hyperparams.vocab_size, model.hyperparams.layer_width):
        return format_into_table_no_position(output, model, vocab, top_row, top_row_label, decoder)
    ############################

    assert output.shape == (model.hyperparams.num_layers, model.hyperparams.num_positions, model.hyperparams.vocab_size, model.hyperparams.layer_width)
    result: List = []
    for n in range(model.hyperparams.num_layers):
        # Label the left-most column with the layer number, and position number
        # The top row has the attention pi weights which do not include position
        result.append([f"layer number: {n}"] + [f"{top_row_label} = {['%.2f' % l.item() for l in top_row[n][:, lw]]}" for lw in range(model.hyperparams.layer_width)])
        for p in range(model.hyperparams.num_positions):
            result.append([f"layer number: {n}, position: {p}"] + [format_prob_vocab(output[n, p, :, lw], vocab) for lw in range(model.hyperparams.layer_width)])

        # Add an empty row between each num_layer
        result.append([None for _ in range(model.hyperparams.layer_width + 1)])
    # Add comments to the bottom of the table:
    if decoder:
        result.append(["Read bottom to top"] + [None for _ in range(model.hyperparams.layer_width)])
    else:
        result.append(["Read top to bottom"] + [None for _ in range(model.hyperparams.layer_width)])
    return result


def format_into_table_no_position(output, model, vocab, top_row, top_row_label: str, decoder: bool = False):
    assert output.shape == (model.hyperparams.num_layers, model.hyperparams.vocab_size, model.hyperparams.layer_width)
    # Takes in the inference output of a model and formats it into a table
    result: List = [[None for _ in range(model.hyperparams.layer_width + 1)] for _ in range(model.hyperparams.num_layers * 4)]
    for n in range(model.hyperparams.num_layers):
        # Layer numbering in the left most column = 0
        # Starts at layer number 0 for the decision/all and for the token
        # Runs until n = num_layers - 1
        result[n * 2][0] = f"layer number: {n}"
        result[n * 2 + 1][0] = f"layer number: {n}"
        # Now fill-in each column of the table
        for lw in range(model.hyperparams.layer_width):
            # Format decoder token weights and activations from the encoder and the decoder
            # Extract the token_pi_encode's input tensor or the token_pi_decode's output tensor
            # We want to look at the n'th num_layer and lw'th column
            # This assumes that the output is of size = (num_layers, vocab_size, layer_width)
            prob_outputs = output[n, :, lw]
            prob_token_pairs_str = format_prob_vocab(prob_outputs, vocab)
            result[n * 2 + 1][lw + 1] = prob_token_pairs_str
            result[n * 2][lw + 1] = f"{top_row_label} = {['%.2f' % l.item() for l in top_row[n][:, lw]]}"
    # Add comments to the bottom of the table:
    empty_row = [None for _ in range(model.hyperparams.layer_width + 1)]
    result.append(empty_row)
    comment: List = [None for _ in range(model.hyperparams.layer_width + 1)]
    if decoder:
        comment[0] = "Read bottom to top"
    else:
        comment[0] = "Read top to bottom"
    result.append(comment)
    return result


def output_to_sheet(result, sheet_name="Sheet1"):
    # Keeping the google sheet imports here so that we can run the code without them
    from google_sheets_api.sheetsstart import Sheet  # type: ignore
    from googleapiclient.errors import HttpError  # type: ignore
    print("****************OUTPUTTING TO GOOGLE SHEET****************")
    print("TAB:", sheet_name)
    spreadsheet_id = "1TpfTFWsFRRyXZU6pEtw1pqEaFpmO9VqXFqxnzpQQfCo"
    creds_dir = pathlib.Path(os.path.dirname(__file__)) / "google_sheets_api"
    sheet = Sheet(path_to_api_creds=creds_dir, sheet_name=sheet_name)

    # Number of maximum retry attempts
    max_attempts = 5
    # Initial delay (in seconds)
    initial_delay = 10
    delay = initial_delay
    # Maximum delay (in seconds)
    max_delay = 160
    for attempt in range(max_attempts):
        try:
            sheet.update_table(spreadsheet_id, sheet.sheet_name, table=result)
            return
        except HttpError as e:
            # Calculate the next delay using exponential backoff
            delay = min(initial_delay * (2 ** attempt), max_delay)

            # Check if it's a rate limit error (HTTP 429)
            if e.resp.status == 429:
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            else:
                print(f"Attempt {attempt + 1} failed. Error: {str(e)}")

            # Sleep for the calculated delay before the next attempt
            time.sleep(delay)
    raise Exception(f"Failed after {max_attempts} attempts.")
