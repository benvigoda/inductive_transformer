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
    try:
        # Pair each prob with it's corresponding token string in the vocab
        prob_token_pairs = [(round(x.item() * 100) / 100, v) for x, v in zip(prob_outputs, vocab)]
        # Sort from highest probability to lowest, so we have the most likely token on top
        prob_token_pairs.sort(key=lambda x: x[0], reverse=True)
        prob_token_pairs_str = "\n".join([str(pw) for pw in prob_token_pairs])
    except Exception as e:
        print("ERROR IN format_prob_vocab:", e)
        print("prob_outputs.shape", prob_outputs.shape)
        print("vocab.shape", len(vocab))
        print("vocab", vocab)
        print("prob_outputs", prob_outputs)
        prob_token_pairs_str = "NA"
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
    return weights  # WARNING: this is a hack to avoid normalizing the weights so the google-sheet really reflects reality
    return custom_normalize(nn.ReLU()(weights), dim=dim)


def send_to_google_sheet(prompt_tensors, preds, truths, token_prob_tensors, model, model_outputs, attention_input, vocab):
    # token_prob_tensors are the training inputs
    prompt_preds = []  # Store the predictions from each prompt_tensor
    model_output_prompts = []  # Store the model outputs from each prompt_tensor
    encoder_attention_pi_weights = torch.stack([
        normalize_weights(model.encoder_layer_0.encoder_attention_pi.weights, dim=0),
        normalize_weights(model.encoder_layer_1.encoder_attention_pi.weights, dim=0),
    ], dim=0)
    encoder_token_pi_weights = torch.stack([
        normalize_weights(model.encoder_layer_0.encoder_token_pi.weights),
        normalize_weights(model.encoder_layer_1.encoder_token_pi.weights),
    ], dim=0)
    encoder_position_pi_weights = torch.stack([
        normalize_weights(model.encoder_layer_0.encoder_position_pi.weights, dim=0),
        normalize_weights(model.encoder_layer_1.encoder_position_pi.weights, dim=0),
    ], dim=0)
    decoder_attention_pi_weights = torch.stack([
        normalize_weights(model.decoder_layer_0.decoder_attention_pi.weights, dim=0),
        normalize_weights(model.decoder_layer_1.decoder_attention_pi.weights, dim=0),
    ], dim=0)
    decoder_token_pi_weights = torch.stack([
        normalize_weights(model.decoder_layer_0.decoder_token_pi.weights),
        normalize_weights(model.decoder_layer_1.decoder_token_pi.weights),
    ], dim=0)
    decoder_position_pi_weights = torch.stack([
        normalize_weights(model.decoder_layer_0.decoder_position_pi.weights, dim=0),
        normalize_weights(model.decoder_layer_1.decoder_position_pi.weights, dim=0),
    ], dim=0)
    for sheet_number, prompt_tensor in enumerate(prompt_tensors):
        # write activations
        model_output_prompt = model(attention_input, prompt_tensor)
        y = model.decoder_pre_output_details  # Actually use the decoder_pre_output_details
        if sheet_number == 0:
            # write decoder weights
            res_decoder_weights = format_into_table(
                output=decoder_token_pi_weights,
                model=model,
                vocab=vocab,
                top_row=decoder_attention_pi_weights,
                top_row_label="decoder attention pi weights",
                decoder=True,
                position_row=decoder_position_pi_weights,
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
                position_row=encoder_position_pi_weights,
            )
            print("encoder weights")
            output_to_sheet(res_encoder_weights, "encoder_weights")

        prompt_preds.append(y)
        model_output_prompts.append(model_output_prompt)
    ############################
    # TODO: We currently do not track the attention_preds and attention_truths
    # Should we?
    # So, for now, we just keep them empty
    attention_preds = [torch.empty(attention_input.shape) for _ in range(len(prompt_tensors))]
    attention_truths = [torch.empty(attention_input.shape) for _ in range(len(prompt_tensors))]
    attention_inputs = [attention_input for _ in range(len(prompt_tensors))]
    ############################
    # With test data sentences running through inference
    # Print the input, pred, and truth (same as input) for each activation:
    res_pred_truth_input = format_into_pred_truth_table(
        model=model,
        vocab=vocab,
        model_outputs=model_output_prompts,
        preds=prompt_preds,
        truths=[truths[i] for i in range(len(prompt_tensors))] if truths is not None else None,
        inputs=prompt_tensors,
        attention_preds=attention_preds,
        attention_truths=attention_truths,
        attention_inputs=attention_inputs,
        title="test prediction versus truth",
    )
    print("inference_input_pred")
    output_to_sheet(res_pred_truth_input, "inference_input_pred")
    attention_preds = [torch.empty(attention_input.shape) for _ in range(len(preds))] if preds is not None else None
    attention_truths = [torch.empty(attention_input.shape) for _ in range(len(preds))] if preds is not None else None
    attention_inputs = [attention_input for _ in range(len(preds))] if preds is not None else None
    if preds is not None and truths is not None and token_prob_tensors is not None:
        res_pred_truth_input = format_into_pred_truth_table(
            model=model,
            vocab=vocab,
            model_outputs=model_outputs,
            preds=preds,
            truths=truths,
            inputs=token_prob_tensors,
            attention_preds=attention_preds,
            attention_truths=attention_truths,
            attention_inputs=attention_inputs,
            title="training prediction versus truth"
        )
        print("training_input_pred_truth")
        output_to_sheet(res_pred_truth_input, "training_input_pred_truth")


def format_into_pred_truth_table(model, vocab, model_outputs, preds, truths, inputs, attention_preds, attention_truths, attention_inputs, title=""):
    # preds, truths, inputs are all size (num_sentences, num_layers, num_positions, vocab_size, layer_width)
    # attention_preds, attention_truths, attention_inputs are all size (num_sentences, layer_width, layer_width)
    # The num_layers index is collapsed as we are only training on the 0th output layer
    num_sentences = len(preds)
    # Initialize the table to None
    table = []

    # Title column
    # For each sentence we have:
    #   For each lw in the layer we have:
    #       For each position we have:
    #           weighted token outputs and attention activation outputs
    # We repeat that for pred, truth, and input
    # We get an empty line between each num_layer
    token_start_index = 1
    attention_start_index = 1 + model.hyperparams.layer_width * model.hyperparams.num_positions
    num_cols = 1 + model.hyperparams.layer_width * model.hyperparams.num_positions + model.hyperparams.layer_width
    row = [None] * num_cols
    row[0] = title
    table.append(row)
    for k in range(num_sentences):
        row = [None] * num_cols
        row[1] = f"sentence {k}"
        table.append(row)
        row = [None] * num_cols
        row[token_start_index] = "token"
        row[attention_start_index] = "attention"
        table.append(row)
        row = [None] * num_cols
        for p_index in range(model.hyperparams.num_positions):
            row[1 + p_index] = format_prob_vocab(model_outputs[k][p_index], vocab)
        table.append(row)
        row = [None] * num_cols
        for n in range(model.hyperparams.num_layers):
            for lw in range(model.hyperparams.layer_width):
                row[token_start_index + lw * model.hyperparams.num_positions] = f"layer_width {lw}"
                row[attention_start_index + lw] = f"layer_width {lw}"
            table.append(row)

            def add_output_to_row(token_output, attention_output, label):
                row = [None] * num_cols
                row[0] = label
                for lw in range(model.hyperparams.layer_width):
                    for p in range(model.hyperparams.num_positions):
                        token_index = token_start_index + lw * model.hyperparams.num_positions + p
                        attention_index = attention_start_index + lw
                        row[token_index] = format_prob_vocab(token_output[k][n][p, :, lw], vocab)
                        if p == 0:  # The attention not split by position, so we just print it once
                            row[attention_index] = format_attention_tensor(attention_output[k][:, lw])
                table.append(row)
                row = [None] * num_cols
            if preds is not None and preds != []:
                add_output_to_row(preds, attention_preds, f"pred layer {n}")
            # if truths is not None and preds != []:  # FIXME
            #     add_output_to_row(truths, attention_truths, f"truth layer {n}")
            if inputs is not None and preds != []:
                add_output_to_row(inputs, attention_inputs, f"input layer {n}")
    return table


def format_into_table(output, model, vocab, top_row, top_row_label: str, decoder: bool = False, position_row: List = []):
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
        # Followed by the position pi weights
        result.append([None] + [f"position weights = {['%.2f' % p.item() for p in position_row[n][:, lw]]}" for lw in range(model.hyperparams.layer_width)])
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
