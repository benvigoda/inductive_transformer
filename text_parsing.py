import argparse
import pathlib
import string
import re
import torch  # type: ignore
from typing import List, Dict

PROBABLE = 1 - 1e-9
IMPROBABLE = 1e-5


class InputData():

    def __init__(self, training_path, inference_path, stop_token='.', print_vals=True):
        self.stop_token = stop_token
        # Reads a text files
        with open(training_path) as f:
            raw_training_text = " ".join(f.readlines())
        with open(inference_path) as f:
            raw_inference_text = " ".join(f.readlines())

        self.raw_training_text = self.clean(raw_training_text)
        self.raw_inference_text = self.clean(raw_inference_text)
        self.text = self.raw_training_text + ' ' + self.raw_inference_text

        # Builds a vocab
        self.vocab: List[str] = self.get_vocab()
        self.vocab_size: int = len(self.vocab)
        self.tokenizer_dict: Dict = self.get_tokenizer()
        self.training_windows = self.stop_token_parsing(text=self.raw_training_text, stop_token=self.stop_token)
        self.inference_windows = self.stop_token_parsing(text=self.raw_inference_text, stop_token=self.stop_token)
        self.window_size = max(len(w) for w in self.training_windows + self.inference_windows)

        # Returns an ordered list of all the words that appear in the file
        if print_vals:
            print('INPUT DATA')
            print(f'vocab_size: {self.vocab_size}')
            print(f'vocab: {self.vocab}')
            print(f'training sentences: {self.training_windows}')
            print(f'inference sentences: {self.inference_windows}')
            print(f'tokenizer_dict: {self.tokenizer_dict}')

    @staticmethod
    def clean(text: str) -> str:
        # To avoid this:
        # ["the", "dog", "bark\nThe", "cat", "meows"] = "the dog barks\nThe cat meows".split()
        # Pad \n's with a space before and after
        clean_text = text.replace("\n", " \n ")
        for punctuation in string.punctuation:
            clean_text = clean_text.replace(punctuation, f" {punctuation} ")
        clean_text = re.sub(' +', ' ', clean_text)  # Remove extra space
        clean_text = clean_text.strip()
        return clean_text

    def get_vocab(self) -> List[str]:
        words = self.text.split()
        vocab = []
        for w in words:
            if w in vocab:
                continue
            vocab.append(w)
        vocab.append("<PADDING>")
        return vocab

    def get_tokenizer(self) -> Dict[str, int]:
        tokenizer_dict = {}
        for i, w in enumerate(self.vocab):
            tokenizer_dict[w] = i
        return tokenizer_dict

    def stop_token_parsing(self, text, stop_token) -> List[List[int]]:
        sentences = text.split(stop_token)  # Split on the stop token
        window_size = max(len(s.split()) for s in sentences)  # Get the max window size as the max number of words in the sentences
        windows = []
        for sentence in sentences:
            if not sentence:
                continue
            words = sentence.split()
            next_window = [self.tokenizer_dict[w] for w in words]
            # Pad the window with -1's (the padding token)
            while len(next_window) < window_size:
                next_window.append(-1)
            windows.append(next_window)
        return windows


class ProbTensors():

    def __init__(self, data: InputData, layer_width: int, print_flag: bool = True):
        self.data = data
        self.layer_width = layer_width
        self.vocab_size = data.vocab_size
        self.num_positions = data.window_size
        self.windows = data.training_windows
        self.improbable = IMPROBABLE
        self.probable = PROBABLE
        self.print_flag = print_flag

        self.attention_input = self.make_attention_input()

    def format_training_data(self, num_layers: int = 1):
        '''
        EXAMPLE:

        2 sentences "the dog. a cat." in the `text_training.txt` file
        vocab: ['the', 'dog', '.', 'cat', 'a', '<PADDING>'],

        sentence 1 "the dog":
        layer 1 left = "the" = (0., -5., -5., -5., -5., -5.)
        layer 0 left  = "dog" = (-5, 0., -5., -5., -5., -5.)
        layer 1 right = NA = (-5., -5., -5., -5., -5., -5.)
        layer 0 right = NA = (-5., -5., -5., -5., -5., -5.)

        sentence 2 "a cat":
        layer 1 left = NA = (-5., -5., -5., -5., -5., -5.)
        layer 0 left = NA = (-5., -5., -5., -5., -5., -5.)
        layer 1 right = "a" = (-5, -5., -5., -5., 0., -5.)
        layer 0 right  = "cat" = (-5, -5., -5., 0., -5., -5.)
        '''
        training_data = []  # A list of tuples of (input, expected_output)
        # Each window corresponds to a sentence. Each sentence is processed individually and appended to the training data
        # We put one sentence per "column" across the layer_width
        for lw, window in enumerate(self.windows):
            if lw >= self.layer_width:
                print(f"WARNING: there are more sentences than layer width. Cannot place more than {self.layer_width} sentences.")
                effective_lw = lw % self.layer_width
            training_output = torch.full((num_layers, self.layer_width, self.vocab_size), self.improbable)
            training_input = torch.full((num_layers, self.layer_width, self.vocab_size), self.improbable)
            # Within a given column, we put one word in each layer
            for word_position_and_layer_num, vocab_index in enumerate(window):
                if word_position_and_layer_num >= num_layers:
                    print(f"WARNING: there are more words in the sentence than num layers. Cannot place more than {num_layers} words.")
                word_prob_tensor = torch.full((self.vocab_size, ), self.improbable)
                word_prob_tensor[vocab_index] = self.probable
                if lw >= self.layer_width:
                    effective_lw = lw % self.layer_width + word_position_and_layer_num
                    training_input[word_position_and_layer_num, effective_lw] = word_prob_tensor
                    training_output[word_position_and_layer_num, effective_lw] = word_prob_tensor
                else:
                    training_input[word_position_and_layer_num, lw] = word_prob_tensor
                    training_output[word_position_and_layer_num, lw] = word_prob_tensor
            training_input = torch.transpose(training_input, 1, 2)
            training_output = torch.transpose(training_output, 1, 2)
            training_output_reshaped = torch.cat([to for to in training_output], dim=0)
            # Add the z_decode_0 output to the training output data
            full_training_output = torch.cat([training_output_reshaped], dim=0)
            assert full_training_output.shape == (num_layers*self.vocab_size, self.layer_width)
            if self.print_flag:
                print(f"word_prob_tensors/training_input in whole model for sentence #{lw + 1}:\n{training_input}")
                print(f"training_input.size():\n{training_input.size()}")
            training_data.append(
                (training_input, full_training_output)
            )
        return training_data

    def make_attention_input(self):
        '''
        For example, in a 2x2 model, we want to make a attention_input that looks like:
        attention_input[i=0, l=0] = 0.5
        attention_input[i=1, l=0] = 0.5
        attention_input[i=0, l=1] = 0.5
        attention_input[i=1, l=1] = 0.5
        '''
        attention_input = torch.full((self.layer_width, self.layer_width), 0.5)
        return attention_input

    def make_inference_prompt_tensors(self, num_layers: int = 1) -> List[torch.Tensor]:
        prompt_tensors = []
        empty_word_prob_tensor = torch.full((self.vocab_size, ), self.improbable)
        empty_layer_prob_tensor = torch.full((self.vocab_size, self.layer_width), float('nan'))
        for window in self.data.inference_windows:
            inference_word_prob_tensor = torch.full((self.vocab_size, ), self.improbable)
            for index_of_probable_word in window:
                inference_word_prob_tensor[index_of_probable_word] = self.probable
            if window[0] == self.data.tokenizer_dict["big"]:
                inference_word_prob_tensor_stacked = torch.stack([
                    torch.stack([inference_word_prob_tensor, empty_word_prob_tensor], dim=1),  # Stack up on layer_width
                    empty_layer_prob_tensor
                ], dim=0)  # Stack up on num_layers
            elif window[0] == self.data.tokenizer_dict["small"]:
                inference_word_prob_tensor_stacked = torch.stack([
                    torch.stack([empty_word_prob_tensor, inference_word_prob_tensor], dim=1),  # Stack up on layer_width
                    empty_layer_prob_tensor,
                ], dim=0)  # Stack up on num_layers
            elif window[0] == self.data.tokenizer_dict["cat"]:
                inference_word_prob_tensor_stacked = torch.stack([
                    empty_layer_prob_tensor,
                    torch.stack([inference_word_prob_tensor, empty_word_prob_tensor], dim=1),  # Stack up on layer_width
                ], dim=0)  # Stack up on num_layers
            elif window[0] == self.data.tokenizer_dict["dog"]:
                inference_word_prob_tensor_stacked = torch.stack([
                    empty_layer_prob_tensor,
                    torch.stack([empty_word_prob_tensor, inference_word_prob_tensor], dim=1),  # Stack up on layer_width
                ], dim=0)  # Stack up on num_layers

            assert inference_word_prob_tensor_stacked.shape == (num_layers, self.vocab_size, self.layer_width)
            prompt_tensors.append(inference_word_prob_tensor_stacked)
        return prompt_tensors


def parse_args():
    parser = argparse.ArgumentParser(description="input text files")
    parser.add_argument("training_text", type=pathlib.Path)  # A text file of sentences to train on
    parser.add_argument("inference_text", type=pathlib.Path)  # A text file of sentences to run inference on
    return parser.parse_args()


def main():
    args = parse_args()
    data = InputData(
        training_path=args.training_text,
        inference_path=args.inference_text,
    )
    print(data)
    prob_tensors = ProbTensors(
        data=data,
        layer_width=4,
    )
    print(prob_tensors)


if __name__ == '__main__':
    main()
