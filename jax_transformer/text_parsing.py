# Copyright 2025 Ben Vigoda, Thomas Rochais, and Erik Strand
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np  # type: ignore
import pathlib
import re
import string
from typing import List, Dict, Tuple
from jax.nn import logsumexp
from jax_transformer.helper_functions import PROBABLE, IMPROBABLE



class InputData:
    def __init__(self, training_path, inference_path, stop_token=".", print_vals=True):
        self.stop_token = stop_token
        # Reads a text files

        if not training_path or not training_path.exists():
            raise FileNotFoundError(f"File {training_path} not found")
        with open(training_path) as f:
            raw_training_text = " ".join(f.readlines()).lower()
        if not inference_path or not inference_path.exists():
            raw_inference_text = ""
        else:
            with open(inference_path) as f:
                raw_inference_text = " ".join(f.readlines()).lower()

        self.raw_training_text = self.clean(raw_training_text)
        self.raw_inference_text = self.clean(raw_inference_text)
        self.text = self.raw_training_text + " " + self.raw_inference_text
        self.training_sentences = [
            sent.strip() for sent in raw_training_text.split(".") if sent.strip()
        ]

        # Builds a vocab
        self.vocab: List[str] = self.get_vocab()
        self.vocab_size: int = len(self.vocab)
        self.tokenizer_dict: Dict = self.get_tokenizer()
        self.training_windows = self.stop_token_parsing(
            text=self.raw_training_text, stop_token=self.stop_token
        )
        self.window_size = self.training_windows.shape[1]
        self.inference_windows = self.stop_token_parsing(
            text=self.raw_inference_text,
            stop_token=self.stop_token,
            min_window_size=self.window_size,
        )
        if len(self.inference_windows) > 0:
            self.window_size = max(self.window_size, self.inference_windows.shape[1])

        # Returns an ordered list of all the words that appear in the file
        if print_vals:
            print("INPUT DATA")
            print(f"vocab_size: {self.vocab_size}")
            print(f"vocab: {self.vocab}")
            print(f"training sentences ({self.training_windows.shape}): {self.training_windows}")
            print(f"inference sentences ({self.inference_windows.shape}): {self.inference_windows}")
            print(f"tokenizer_dict: {self.tokenizer_dict}")

    @property
    def padding_token(self) -> int:
        return self.vocab_size - 1

    @staticmethod
    def clean(text: str) -> str:
        # To avoid this:
        # ["the", "dog", "bark\nThe", "cat", "meows"] = "the dog barks\nThe cat meows".split()
        # Pad \n's with a space before and after
        clean_text = text.replace("\n", " \n ")
        for punctuation in string.punctuation:
            clean_text = clean_text.replace(punctuation, f" {punctuation} ")
        clean_text = re.sub(" +", " ", clean_text)  # Remove extra space
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

    def stop_token_parsing(self, text, stop_token, min_window_size=0) -> jax.Array:
        sentences = text.split(stop_token)  # Split on the stop token
        window_size = max(
            len(s.split()) for s in sentences
        )  # Get the max window size as the max number of words in the sentences
        window_size = max(window_size, min_window_size)
        windows = []
        for sentence in sentences:
            if not sentence:
                continue
            words = sentence.split()
            next_window = [self.tokenizer_dict[w] for w in words]
            # Add padding tokens to fill the rest of the window.
            while len(next_window) < window_size:
                next_window.append(self.padding_token)
            windows.append(next_window)
        windows = jnp.array(windows)  # type: ignore
        return windows  # type: ignore


class ProbTensors:
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

    def format_training_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
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
        """
        training_data = []  # A list of tuples of (input, expected_output)
        # Each window corresponds to a sentence. Each sentence is processed individually and
        # appended to the training data. We train on the expected_output to be the same as the
        # input. self.windows is an array of vocab indices, shape (num_sentences, num_words). For
        # example, the first window is [0, 1] corresponding to "small dog".
        for window in self.windows:
            # Convert the window to a one-hot encoding
            # Note: we could use https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.one_hot.html
            output_tensor = np.full(
                (self.num_positions, self.vocab_size), self.improbable
            )
            for word_position, vocab_index in enumerate(window):
                output_tensor[word_position, vocab_index] = self.probable
            # Reshape the training element to be (1, num_positions, vocab_size, 1)
            training_element = output_tensor[None, :, :, None]
            # Make copies along the num_layer and layer_width dimensions
            input_tensor = np.broadcast_to(
                training_element,
                (
                    self.num_layers,
                    self.num_positions,
                    self.vocab_size,
                    self.layer_width,
                ),
            )

            # epsilon = 1e-4
            # if self.layer_width == 2:
            #     # column-specific bias  [+ε, −ε]
            #     bias = np.array([+epsilon, -epsilon], dtype=np.float32)    # shape (layer_width,)
            # else:
            #     # give each column its own small bias in (−ε , +ε)
            #     rng  = np.random.default_rng()
            #     bias = rng.uniform(-epsilon, +epsilon,
            #                        size=(self.layer_width,)).astype(np.float32)
            # input_tensor = input_tensor + bias                # broadcasts over the first 3 axes

            # re-normalise so each (layer, position, column) slice is still a valid log-prob distribution
            # input_tensor = input_tensor - logsumexp(input_tensor, axis=2, keepdims=True)


            if self.print_flag:
                print(f"format_training_data for window {window}:\n{input_tensor}")
                print(f"input_tensor.size:\n{input_tensor.size}")
            training_data.append((input_tensor, output_tensor))
        return training_data

    def make_attention_input(self):
        """
        For example, in a 2x2 model, we want to make a attention_input that looks like:

        attention_input[i=0, l=0] = log(0.5)
        attention_input[i=1, l=0] = log(0.5)
        attention_input[i=0, l=1] = log(0.5)

        """

        attention_input = np.full((2, self.layer_width), jnp.log(0.5))  # A bernoulli input
        # attention_input[1, 0] = 0.5
        # attention_input[1, 1] = 0.5
        return attention_input

    def make_inference_prompt_tensors(self) -> List[np.ndarray]:
        inference_data = []  # A list of tuples of (input, expected_output)
        # We train on the expected_output to be the same as the input
        for (
            window
        ) in (
            self.data.inference_windows
        ):  # self.windows is a list of lists of vocab indices
            # For example, the first window is [0, 1] corresponding to "small dog"
            inference_element = np.full(
                (self.num_positions, self.vocab_size), self.improbable
            )
            for word_position, vocab_index in enumerate(window):
                inference_element[word_position, vocab_index] = self.probable
            # Reshape the training element to be (1, num_positions, vocab_size, 1)
            inference_element = inference_element[None, :, :, None]

            # Make copies along the num_layer and layer_width dimensions
            input_tensor = np.broadcast_to(
                inference_element,
                (
                    self.num_layers,
                    self.num_positions,
                    self.vocab_size,
                    self.layer_width,
                ),
            )


            inference_data.append(input_tensor)
        return inference_data


def parse_args():
    parser = argparse.ArgumentParser(description="input text files")
    parser.add_argument(
        "training_text", type=pathlib.Path
    )  # A text file of sentences to train on
    parser.add_argument(
        "inference_text", type=pathlib.Path
    )  # A text file of sentences to run inference on
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


if __name__ == "__main__":
    main()
