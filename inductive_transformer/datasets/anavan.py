from itertools import product
from typing import List, Optional
import click
import numpy as np


class ANAVAN:
    """Generates 6 word sentences of the form adjective noun adverb verb adjective noun."""

    def __init__(self, left_zeroth, left_first, left_second, left_third, left_fourth, left_fifth, right_zeroth,
                 right_first, right_second, right_third, right_fourth, right_fifth):
        self.left_zeroth_words = left_zeroth
        self.left_first_words = left_first
        self.left_second_words = left_second
        self.left_third_words = left_third
        self.left_fourth_words = left_fourth
        self.left_fifth_words = left_fifth
        self.right_zeroth_words = right_zeroth
        self.right_first_words = right_first
        self.right_second_words = right_second
        self.right_third_words = right_third
        self.right_fourth_words = right_fourth
        self.right_fifth_words = right_fifth

    def clear_right_words(self):
        self.right_zeroth_words = []
        self.right_first_words = []
        self.right_second_words = []
        self.right_third_words = []
        self.right_fourth_words = []
        self.right_fifth_words = []

    def clear_left_words(self):
        self.left_zeroth_words = []
        self.left_first_words = []
        self.left_second_words = []
        self.left_third_words = []
        self.left_fourth_words = []
        self.left_fifth_words = []

    def get_valid_left_ordered_words(self):
        return [
            self.left_zeroth_words,
            self.left_first_words,
            self.left_second_words,
            self.left_third_words,
            self.left_fourth_words,
            self.left_fifth_words,
        ]

    def get_valid_right_ordered_words(self):
        return [
            self.right_zeroth_words,
            self.right_first_words,
            self.right_second_words,
            self.right_third_words,
            self.right_fourth_words,
            self.right_fifth_words,
        ]

    def generate(
        self,
        num_sentences: Optional[int] = None,
        side: str = 'both',
        single_synonyms: Optional[List[int]] = None
    ) -> List[str]:
        sentences = []

        if single_synonyms is None:
            single_synonyms = []

        def generate_side(words_lists: List[List[str]], single_positions: List[int]) -> List[str]:
            # Generate all combinations for the given side
            # Adjust lists for single synonyms
            adjusted_lists = []
            for idx, word_list in enumerate(words_lists):
                if idx in single_positions:
                    adjusted_lists.append([word_list[0]])  # Use only the first synonym
                else:
                    adjusted_lists.append(word_list)
            return [' '.join(sentence) for sentence in product(*adjusted_lists)]

        if side in ['left', 'both']:
            left_lists = self.get_valid_left_ordered_words()
            left_single_positions = [i for i in range(len(left_lists)) if i in single_synonyms]
            left_sentences = generate_side(left_lists, left_single_positions)
            sentences.extend(left_sentences)

        if side in ['right', 'both']:
            right_lists = self.get_valid_right_ordered_words()
            right_single_positions = [i for i in range(len(right_lists)) if i in single_synonyms]
            right_sentences = generate_side(right_lists, right_single_positions)
            sentences.extend(right_sentences)

        # Limit the number of sentences if necessary
        if num_sentences is None:
            return sentences
        else:
            num_sentences = min(num_sentences, len(sentences))
            return sentences[:num_sentences]

    def is_valid(self, sentence):
        num_words = 6
        words = sentence.lower().split()

        if len(words) != num_words:
            return False

        if words[0] in self.left_zeroth_words:
            if words[1] not in self.left_first_words:
                return False
            if words[2] not in self.left_second_words:
                return False
            if words[3] not in self.left_third_words:
                return False
            if words[4] not in self.left_fourth_words:
                return False
            if words[5] not in self.left_fifth_words:
                return False
            return True
        elif words[0] in self.right_zeroth_words:
            if words[1] not in self.right_first_words:
                return False
            if words[2] not in self.right_second_words:
                return False
            if words[3] not in self.right_third_words:
                return False
            if words[4] not in self.right_fourth_words:
                return False
            if words[5] not in self.right_fifth_words:
                return False
            return True
        else:
            return False


def make_cat_dog_anavan():
    valid_left_zeroth_words = [
        "small",
        "little",
        "tiny",
        "micro",
        "mini",
        "pico",
        "femto",
        "diminimus",
        # "itty",
        # "teenyweeny",
    ]
    valid_right_zeroth_words = [
        "extralarge",
        "gargantuan",
        "large",
        "giant",
        "huge",
        "humongous",
        "enormous",
        "big",
    ]
    valid_left_first_words = [
        "dogs",
        "canines",
    ]
    valid_right_first_words = [
        "cats",
        "felines",
    ]
    valid_left_second_words = [
        "often",
        "usually",
        "commonly",
        "frequently",
    ]
    valid_right_second_words = [
        "sometimes",
        "occasionally",
        "rarely",
        "never",
    ]
    valid_left_third_words = [
        "fear",
        "avoid",
    ]
    valid_right_third_words = [
        "chase",
        "intimidate",
        "eat",
    ]

    return ANAVAN(
        left_zeroth=valid_left_zeroth_words,
        left_first=valid_left_first_words,
        left_second=valid_left_second_words,
        left_third=valid_left_third_words,
        left_fourth=valid_right_zeroth_words,
        left_fifth=valid_right_first_words,
        right_zeroth=valid_right_zeroth_words,
        right_first=valid_right_first_words,
        right_second=valid_right_second_words,
        right_third=valid_right_third_words,
        right_fourth=valid_left_zeroth_words,
        right_fifth=valid_left_first_words,
    )


def make_cat_dog_worm_bird_anavan():
    valid_left_zeroth_words = [
        "small",
        "little",
        "tiny",
        "micro",
        "mini",
        "pico",
        "femto",
        "diminimus",
    ]
    valid_left_first_words = [
        "dogs",
        "canines",
    ]
    valid_left_second_words = [
        "often",
        "usually",
        "commonly",
        "frequently",
    ]
    valid_left_third_words = [
        "fear",
        "avoid",
    ]
    valid_left_fourth_words = [
        "extralarge",
        "gargantuan",
        "large",
        "giant",
        "huge",
        "humongous",
        "enormous",
        "big",
    ]
    valid_left_fifth_words = [
        "cats",
        "felines"
    ]

    valid_right_zeroth_words = [
        "wriggley",
        "gross",
        "slimy",
        "disgusting",
        "icky",
        "lousy",
        "juicy",
        "squishy",
    ]
    valid_right_first_words = [
        "worms",
        "earthworms"
    ]
    valid_right_second_words = [
        "sometimes",
        "occasionally",
        "rarely",
        "never",
    ]
    valid_right_third_words = [
        "chase",
        "intimidate",
    ]
    valid_right_fourth_words = [
        "angry",
        "hateful",
        "mean",
        "nasty",
        "unpleasant",
        "vicious",
        "violent",
        "wicked",
    ]
    valid_right_fifth_words = [
        "birds",
        "avians",
    ]

    return ANAVAN(
        left_zeroth=valid_left_zeroth_words,
        left_first=valid_left_first_words,
        left_second=valid_left_second_words,
        left_third=valid_left_third_words,
        left_fourth=valid_left_fourth_words,
        left_fifth=valid_left_fifth_words,
        right_zeroth=valid_right_zeroth_words,
        right_first=valid_right_first_words,
        right_second=valid_right_second_words,
        right_third=valid_right_third_words,
        right_fourth=valid_right_fourth_words,
        right_fifth=valid_right_fifth_words,
    )


@click.command()
@click.option("--out_file", "-o", type=click.Path(), default=None)
@click.option("--num_sentences", "-n", type=int, default=10)
@click.option("--seed", "-s", type=int, default=1889567120394717)
def main(out_file, num_sentences, seed):
    grammar = make_cat_dog_worm_bird_anavan()
    sentences = grammar.generate()
    print(f"Generated {len(sentences)} valid sentences.")

    print(f"Shuffling sentences with seed {seed}")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(sentences))
    rng.shuffle(indices)

    num_sentences = min(num_sentences, len(sentences))
    indices = indices[:num_sentences]
    sentences = [sentences[idx] for idx in indices]

    if out_file is None:
        out_file = f"anavan_{num_sentences}.txt"

    print(f"Writing {num_sentences} sentences to {out_file}")
    with open(out_file, "w") as f:
        for sentence in sentences:
            f.write(sentence + '\n')


if __name__ == "__main__":
    main()
