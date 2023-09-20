import torch  # type: ignore
import pathlib
import os
import time
from google_sheets_api.sheetsstart import Sheet
from googleapiclient.errors import HttpError  # type: ignore


def format_prob_vocab(log_prob_outputs, vocab):
    """
    Takes in a log_prob_outputs tensor and a list of vocab words
    It assumes the two are sorted in corresponding order
    (meaning) that the first entry in the log_prob_outputs tensors
    does correspond to the first word in the vocab list

    Return a string corresponding
    to the list of formatted tuples linking each prob to it's corresponding
    word, sorted from highest prob to lowest
    """
    # Pair each prob with it's corresponding word string in the vocab
    prob_word_pairs = [(round(x.item() * 100) / 100, v) for x, v in zip(log_prob_outputs, vocab)]
    # Sort from highest probability to lowest, so we have the most likely word on top
    prob_word_pairs.sort(key=lambda x: x[0], reverse=True)
    prob_word_pairs_str = "\n".join([str(pw) for pw in prob_word_pairs])
    return prob_word_pairs_str


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
    print('***************')
    # print("logical_encoder.decision_X1LSE.weights:", [layer.decision_X1LSE.weights for layer in model.logical_encoder.layers])
    # print("logical_encoder.word_X1LSE.weights:", [layer.word_X1LSE.weights for layer in model.logical_encoder.layers])
    print("logical_decoder.concept_ALLSUM_decode.weights", [layer.concept_ALLSUM_decode.weights for layer in model.logical_decoder.layers])
    # print("logical_decoder.word_X1LSE_decode_layer.weights", [decoder_layer.word_X1LSE_decode_layer.weights for decoder_layer in model.logical_decoder.layers])
    print('---------------')
    # [decoder_layer.printable() for decoder_layer in model.logical_decoder.layers]


def send_to_google_sheet(prompt_tensors, preds, truths, word_log_prob_tensors, model, input_decision_activations, vocab):
    # word_log_prob_tensors are the training inputs
    prompt_preds = []  # Store the predictions from each prompt_tensor
    encoder_attention_pi_weights = torch.FloatTensor([model.encoder_layer_0.attention.weights, model.encoder_layer_1.attention.weights])
    encoder_word_pi_weights = torch.FloatTensor([model.encoder_layer_0.word.weights, model.encoder_layer_1.word.weights])
    decoder_attention_pi_weights = torch.FloatTensor([model.decoder_layer_0.attention.weights, model.decoder_layer_1.attention.weights])
    decoder_word_pi_weights = torch.FloatTensor([model.decoder_layer_0.word.weights, model.decoder_layer_1.word.weights])
    for sheet_number, prompt_tensor in enumerate(prompt_tensors):
        # write activations
        y = model(prompt_tensor, input_decision_activations)

        if sheet_number == 0:

            # write decoder weights
            res_decoder_weights = format_into_table(
                output=decoder_word_pi_weights,
                model=model,
                vocab=vocab,
                top_row=decoder_attention_pi_weights,
                top_row_label="decoder attention pi weights",
                decoder=True,
            )
            output_to_sheet(res_decoder_weights, "decoder_weights")

            # write encoder weights
            # The encoder weights are (vocab_size, layer_width) while the decoder weights are (layer_width, vocab_size)
            # For print consistency in the google sheet, we need to transpose the encoder weights
            res_encoder_weights = format_into_table(
                # TODO: We previously had to transpose "encoder_word_pi_weights" and "encoder_attention_pi_weights".
                # Do we still need to do that?
                output=encoder_word_pi_weights,
                model=model,
                vocab=vocab,
                top_row=encoder_attention_pi_weights,
                top_row_label="encoder attention pi weights",
                decoder=False,
            )
            output_to_sheet(res_encoder_weights, "encoder_weights")

        prompt_preds.append(y)
    # With test data sentences running through inference
    # Print the input, pred, and truth (same as input) for each activation:
    res_pred_truth_input = format_into_pred_truth_table(
        model=model,
        vocab=vocab,
        preds=prompt_preds,
        # The input that goes into the encoder is transposed in its last two entries compared to the decoder. For printing purposes here, we need to transpose it back to the same format
        # TODO: is that still true???
        truths=torch.transpose(prompt_tensors, 2, 3),
        inputs=torch.transpose(prompt_tensors, 2, 3),
        title="test prediction versus truth"
    )
    output_to_sheet(res_pred_truth_input, "inference_input_pred")

    if preds is not None and truths is not None and word_log_prob_tensors is not None:
        # With the last training data sentence running through inference
        # Print the input, pred, and truth to google-sheet
        res_pred_truth_input = format_into_pred_truth_table(
            model=model,
            vocab=vocab,
            preds=preds,
            truths=truths,
            # The input that goes into the encoder is transposed in its last two entries compared to the decoder. For printing purposes here, we need to transpose it back to the same format
            # TODO: is that still true???
            inputs=torch.transpose(word_log_prob_tensors, 2, 3),
            title="training prediction versus truth"
        )
        output_to_sheet(res_pred_truth_input, "training_input_pred_truth")


def format_into_pred_truth_table(model, vocab, preds, truths, inputs, title=""):
    # preds, truths, inputs are all size (num_sentences, num_layers, layer_width, vocab_size)

    # Initialize the table to None
    table = [[None for _ in range(1 + len(preds) * (model.hyperparameters.layer_width + 1))] for _ in range(model.hyperparameters.num_layers * 4 + 1)]
    # Title in the top left corner
    table[0][0] = title

    # Label each sentence
    for k in range(len(preds)):
        # sentence 0 is in column 1
        # sentence 1 is in column 1 + layer_width + 1
        # sentence 2 is in column 1 + layer_width + 1 + layer_width + 1
        # ...
        table[0][1 + k * (model.hyperparameters.layer_width + 1)] = f"sentence {k}"

    # Fill-in the word-lists
    for n in range(model.hyperparameters.num_layers):
        table[1 + n * 4][0] = f"pred layer {n}"
        table[1 + n * 4 + 1][0] = f"truth layer {n}"
        table[1 + n * 4 + 2][0] = f"input layer {n}"
        # Leave the 4th line blank
        for lw in range(model.hyperparameters.layer_width):
            for k in range(len(preds)):
                # row = n * 4
                # for sentence 0: columns are 1 + lw
                # for sentence 1: columns are 1 + layer_width + 1 + lw
                # for sentence 2: columns are 1 + layer_width + 1 + num_layers + 1 + lw
                # ...

                table[1 + n * 4][(1 + model.hyperparameters.layer_width) * k + 1 + lw] = format_prob_vocab(preds[k][n][lw], vocab)
                table[1 + n * 4 + 1][(1 + model.hyperparameters.layer_width) * k + 1 + lw] = format_prob_vocab(truths[k][n][lw], vocab)
                table[1 + n * 4 + 2][(1 + model.hyperparameters.layer_width) * k + 1 + lw] = format_prob_vocab(inputs[k][n][lw], vocab)

    return table


def format_into_table(output, model, vocab, top_row, top_row_label: str, decoder: bool = False):
    # Takes in the inference output of a model and formats it into a table
    result = [[None for _ in range(model.hyperparameters.layer_width + 1)] for _ in range(model.hyperparameters.num_layers * 4)]

    for n in range(model.hyperparameters.num_layers):

        # Layer numbering in the left most column = 0
        # Starts at layer number 0 for the decision/all and for the word
        # Runs until n = num_layers - 1
        result[n * 2][0] = f"layer number: {n}"
        result[n * 2 + 1][0] = f"layer number: {n}"

        # Now fill-in each column of the table
        for lw in range(model.hyperparameters.layer_width):
            # Format decoder word weights and activations from the encoder and the decoder
            # Extract the word_X1LSE_encode's input tensor or the word_X1LSE_decode's output tensor
            # We want to look at the n'th num_layer and lw'th column
            # This assumes that the output is of size = (num_layers, layer_width, vocab_size)
            log_prob_outputs = output[n, lw]
            prob_word_pairs_str = format_prob_vocab(log_prob_outputs, vocab)

            # Format decision weights and activations from the encoder and decoder
            # For the decoder we print the ALLSUM weights first
            # For both the encoder and decoder, we index in a transposed way.
            # Namely, we print decision_X1LSE_w, z_decode_w or concept_ALLSUM_decode_w[:, lw] in the column lw
            if decoder:
                result[n * 2][lw + 1] = prob_word_pairs_str
                result[n * 2 + 1][lw + 1] = f"{top_row_label} = {['%.2f' % l.item() for l in top_row[n][:, lw]]}"
            else:
                result[n * 2 + 1][lw + 1] = prob_word_pairs_str
                # here we do top_row[n][:, lw] the same as above in the decoder, because
                # the decision_weights question we want to answer for ourselves in the google sheet is
                # "which concept has this encoder X1LSE learned to listen to"
                # the decision_activations question we want to answer for ourselves in the google sheet is
                # "which concept IS this encoder X1LSE listening to when we input a particular sentence"
                result[n * 2][lw + 1] = f"{top_row_label} = {['%.2f' % l.item() for l in top_row[n][:, lw]]}"

    # Add comments to the bottom of the table:
    empty_row = [None for _ in range(model.hyperparameters.layer_width + 1)]
    result.append(empty_row)
    comment = [None for _ in range(model.hyperparameters.layer_width + 1)]
    if decoder:
        comment[0] = "Read bottom to top"
    else:
        comment[0] = "Read top to bottom"
    result.append(comment)
    return result


def output_to_sheet(result, sheet_name="Sheet1"):
    print("****************OUTPUTTING TO GOOGLE SHEET****************")
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
