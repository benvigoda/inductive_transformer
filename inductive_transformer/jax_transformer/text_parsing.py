import argparse
import numpy as np
import pathlib
import re
import string
from typing import List, Dict, Tuple

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
        self.num_layers = data.window_size
        self.windows = data.training_windows
        self.improbable = IMPROBABLE
        self.probable = PROBABLE
        self.print_flag = print_flag

        self.attention_input = self.make_attention_input()

    def format_training_data(self, num_layers: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
        '''
        EXAMPLE INPUT DATA:
        2 sentences "small dog. big cat." in the `text_training.txt` file
        vocab: ['small', 'dog', '.', 'big', 'cat', '<PADDING>'],

        Input data size = (num_positions, vocab_size, layer_width)
        However we will have the same values for every value of lw (which indexes layer_width),
        so our example is just size = (num_positions=2, vocab_size=5)
        We will also have the same values in every layer.

       FOR THE DECODER AND ENCODER
        "small dog" = [0, 1]
        [
            (0., -5., -5., -5., -5., -5.)
            (-5., 0., -5., -5., -5., -5.)
        ]
        "big cat" = [3, 4]
        [
            (-5., -5., -5., 0., -5., -5.)
            (-5., -5., -5., -5., 0., -5.)
        ]

        Repeat this same data for every l (indexing layer) and lw (indexing layer width)
        '''
        training_data = []  # A list of tuples of (input, expected_output)
        # Each window corresponds to a sentence. Each sentence is processed individually and appended to the training data
        # We train on the expected_output to be the same as the input
        for window in self.windows:  # self.windows is a list of lists of vocab indices
            # For example, the first window is [0, 1] corresponding to "small dog"
            output_tensor = np.full((self.num_positions, self.vocab_size), self.improbable)
            for word_position, vocab_index in enumerate(window):
                output_tensor[word_position, vocab_index] = self.probable
            # Reshape the training element to be (1, num_positions, vocab_size, 1)
            training_element = output_tensor[None, :, :, None]
            # Make copies along the num_layer and layer_width dimensions
            input_tensor = np.broadcast_to(training_element, (self.num_layers, self.num_positions, self.vocab_size, self.layer_width))
            if self.print_flag:
                print(f"format_training_data for window {window}:\n{input_tensor}")
                print(f"input_tensor.size:\n{input_tensor.size}")
            training_data.append(
                (input_tensor, output_tensor)
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
        attention_input = np.full((2, self.layer_width), 0.5)  # A bernoulli input
        return attention_input

    def make_inference_prompt_tensors(self, num_layers: int = 1) -> List[np.ndarray]:
        inference_data = []  # A list of tuples of (input, expected_output)
        # We train on the expected_output to be the same as the input
        for window in self.data.inference_windows:  # self.windows is a list of lists of vocab indices
            # For example, the first window is [0, 1] corresponding to "small dog"
            inference_element = np.full((self.num_positions, self.vocab_size), self.improbable)
            for word_position, vocab_index in enumerate(window):
                inference_element[word_position, vocab_index] = self.probable
            # Reshape the training element to be (1, num_positions, vocab_size, 1)
            inference_element = inference_element[None, :, :, None]
            # Make copies along the num_layer and layer_width dimensions
            input_tensor = np.broadcast_to(inference_element, (self.num_layers, self.num_positions, self.vocab_size, self.layer_width))
            inference_data.append(
                input_tensor
            )
        return inference_data


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
