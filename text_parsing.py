
import string
import re
from typing import List, Dict


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

        # Builds a vocab
        self.vocab: List[str] = self.get_vocab()
        self.vocab_size: int = len(self.vocab)
        self.tokenizer_dict: Dict = self.get_tokenizer()
        self.windows = self.stop_token_parsing(stop_token)

        # Returns an ordered list of all the words that appear in the file
        if print_vals:
            print('INPUT DATA')
            print(f'vocab_size: {self.vocab_size}')
            print(f'vocab: {self.vocab}')
            print(f'sentences: {self.windows}')
            print(f'tokenizer_dict: {self.tokenizer_dict}')

    @staticmethod
    def clean(text: str):
        # To avoid this:
        # ["the", "dog", "bark\nThe", "cat", "meows"] = "the dog barks\nThe cat meows".split()
        # Pad \n's with a space before and after
        clean_text = text.replace("\n", " \n ")
        for punctuation in string.punctuation:
            clean_text = clean_text.replace(punctuation, f" {punctuation} ")
        clean_text = re.sub(' +', ' ', clean_text)  # Remove extra space
        clean_text = clean_text.strip()
        return clean_text

    def get_vocab(self):
        words = self.text.split()
        vocab = []
        for w in words:
            if w in vocab:
                continue
            vocab.append(w)
        vocab.append("<PADDING>")
        return vocab

    def get_tokenizer(self):
        tokenizer_dict = {}
        for i, w in enumerate(self.vocab):
            tokenizer_dict[w] = i
        return tokenizer_dict

    def stop_token_parsing(self, stop_token):
        sentences = self.text.split(stop_token)
        self.window_size = max(len(s.split()) for s in sentences)  # Warning: overwrite the window_size
        windows = []
        for sentence in sentences:
            if not sentence:
                continue
            words = sentence.split()
            next_window = [self.tokenizer_dict[w] for w in words]
            while len(next_window) < self.window_size:
                next_window.append(-1)
            windows.append(next_window)
        return windows

    def word_to_word_index(self, word):
        word_index = self.vocab.index(word)
        return word_index
