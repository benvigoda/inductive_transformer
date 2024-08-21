from typing import List, Optional
from dataclasses import dataclass
from itertools import product


@dataclass
class Synonyms:
    valid_left_zeroth_words = [
        "small",
        # "little",
        # "tiny",
        # "micro",
        # "mini",
        # "pico",
        # "femto",
        # "diminimus",
    ]
    valid_left_first_words = [
        "dogs",
        # "canines",
    ]
    valid_left_second_words = [
        "often",
        # "usually",
        # "commonly",
        # "frequently",
    ]
    valid_left_third_words = [
        "fear",
        # "avoid",
    ]
    valid_left_fourth_words = [
        "extralarge",
        # "gargantuan",
        # "large",
        # "giant",
        # "huge",
        # "humongous",
        # "enormous",
        # "big",
    ]
    valid_left_fifth_words = [
        "cats",
        "felines"
    ]

    valid_right_zeroth_words = [
        "wriggley",
        # "gross",
        # "slimy",
        # "disgusting",
        # "icky",
        # "lousy",
        # "juicy",
        # "squishy",
    ]
    valid_right_first_words = [
        "worms",
        # "earthworms"
    ]
    valid_right_second_words = [
        "sometimes",
        # "occasionally",
        # "rarely",
        # "never",
    ]
    valid_right_third_words = [
        "chase",
        # "intimidate",
    ]
    valid_right_fourth_words = [
        "angry",
        # "hateful",
        # "mean",
        # "nasty",
        # "unpleasant",
        # "vicious",
        # "violent",
        # "wicked",
    ]
    valid_right_fifth_words = [
        "birds",
        "avians",
    ]

    def cats_and_dogs_overwrite(self):
        # Use a different synonym set
        self.valid_left_zeroth_words = [
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
        self.valid_right_zeroth_words = [
            "extralarge",
            "gargantuan",
            "large",
            "giant",
            "huge",
            "humongous",
            "enormous",
            "big",
        ]
        self.valid_left_first_words = [
            "dogs",
            "canines",
        ]
        self.valid_right_first_words = [
            "cats",
            "felines",
        ]
        self.valid_left_second_words = [
            "often",
            "usually",
            "commonly",
            "frequently",
        ]
        self.valid_right_second_words = [
            "sometimes",
            "occasionally",
            "rarely",
            "never",
        ]
        self.valid_left_third_words = [
            "fear",
            "avoid",
        ]
        self.valid_right_third_words = [
            "chase",
            "intimidate",
            "eat",
        ]

        self.valid_left_fourth_words = self.valid_right_zeroth_words
        self.valid_right_fourth_words = self.valid_left_zeroth_words
        self.valid_left_fifth_words = self.valid_right_first_words
        self.valid_right_fifth_words = self.valid_left_first_words

    def zero_right_words(self):
        self.valid_right_zeroth_words = []
        self.valid_right_first_words = []
        self.valid_right_second_words = []
        self.valid_right_third_words = []
        self.valid_right_fourth_words = []
        self.valid_right_fifth_words = []

    def zero_left_words(self):
        self.valid_left_zeroth_words = []
        self.valid_left_first_words = []
        self.valid_left_second_words = []
        self.valid_left_third_words = []
        self.valid_left_fourth_words = []
        self.valid_left_fifth_words = []

    def get_valid_left_ordered_words(self):
        return [
            self.valid_left_zeroth_words,
            self.valid_left_first_words,
            self.valid_left_second_words,
            self.valid_left_third_words,
            self.valid_left_fourth_words,
            self.valid_left_fifth_words,
        ]

    def get_valid_right_ordered_words(self):
        return [
            self.valid_right_zeroth_words,
            self.valid_right_first_words,
            self.valid_right_second_words,
            self.valid_right_third_words,
            self.valid_right_fourth_words,
            self.valid_right_fifth_words,
        ]

    def get_valid_pairs(self):
        valid_pairs = {
            (0, 1): {
                (a, b)
                for a in self.valid_left_zeroth_words
                for b in self.valid_left_first_words
            } | {
                (a, b)
                for a in self.valid_right_zeroth_words
                for b in self.valid_right_first_words
            },
            (1, 3): {
                (a, b)
                for a in self.valid_left_first_words
                for b in self.valid_left_third_words
            } | {
                (a, b)
                for a in self.valid_right_first_words
                for b in self.valid_right_third_words
            },
            (2, 3): {
                (a, b)
                for a in self.valid_left_second_words
                for b in self.valid_left_third_words
            } | {
                (a, b)
                for a in self.valid_right_second_words
                for b in self.valid_right_third_words
            },
            (3, 5): {
                (a, b)
                for a in self.valid_left_third_words
                for b in self.valid_left_fifth_words
            } | {
                (a, b)
                for a in self.valid_right_third_words
                for b in self.valid_right_fifth_words
            },
            (4, 5): {
                (a, b)
                for a in self.valid_left_fourth_words
                for b in self.valid_left_fifth_words
            } | {
                (a, b)
                for a in self.valid_right_fourth_words
                for b in self.valid_right_fifth_words
            },
        }
        return valid_pairs

    # valid_pairs = {
    #     (0, 1): {
    #         ("small", "dogs"), ("small", "canines"),
    #         ("little", "dogs"), ("little", "canines"),
    #         ("tiny", "dogs"), ("tiny", "canines"),
    #         ("micro", "dogs"), ("micro", "canines"),
    #         ("mini", "dogs"), ("mini", "canines"),
    #         ("pico", "dogs"), ("pico", "canines"),
    #         ("femto", "dogs"), ("femto", "canines"),
    #         ("diminimus", "dogs"), ("diminimus", "canines"),
    #         ("itty", "dogs"), ("itty", "canines"),
    #         ("teenyweeny", "dogs"), ("teenyweeny", "canines"),
    #         ("extralarge", "cats"), ("extralarge", "felines"),
    #         ("gargantuan", "cats"), ("gargantuan", "felines"),
    #         ("large", "cats"), ("large", "felines"),
    #         ("giant", "cats"), ("giant", "felines"),
    #         ("huge", "cats"), ("huge", "felines"),
    #         ("humongous", "cats"), ("humongous", "felines"),
    #         ("enormous", "cats"), ("enormous", "felines"),
    #         ("big", "cats"), ("big", "felines"),
    #     },
    #     (1, 3): {
    #         ("dogs", "fear"), ("dogs", "avoid"),
    #         ("canines", "fear"), ("canines", "avoid"),
    #         ("cats", "chase"), ("cats", "intimidate"), ("cats", "eat"),
    #         ("felines", "chase"), ("felines", "intimidate"), ("felines", "eat"),
    #     },
    #     (2, 3): {
    #         ("often", "fear"), ("often", "avoid"),
    #         ("usually", "fear"), ("usually", "avoid"),
    #         ("commonly", "fear"), ("commonly", "avoid"),
    #         ("frequently", "fear"), ("frequently", "avoid"),
    #         ("sometimes", "chase"), ("sometimes", "intimidate"), ("sometimes", "eat"),
    #         ("occasionally", "chase"), ("occasionally", "intimidate"), ("occasionally", "eat"),
    #         ("rarely", "fear"), ("rarely", "avoid"),
    #         ("never", "fear"), ("never", "avoid")
    #     },
    #     (3, 5): {
    #         ("fear", "cats"), ("fear", "felines"),
    #         ("avoid", "cats"), ("avoid", "felines"),
    #         ("chase", "dogs"), ("chase", "canines"),
    #         ("intimidate", "dogs"), ("intimidate", "canines"),
    #         ("eat", "dogs"), ("eat", "canines"),
    #     },
    #     (4, 5): {
    #         ("large", "cats"), ("large", "felines"),
    #         ("giant", "cats"), ("giant", "felines"),
    #         ("huge", "cats"), ("huge", "felines"),
    #         ("humongous", "cats"), ("humongous", "felines"),
    #         ("enormous", "cats"), ("enormous", "felines"),
    #         ("big", "cats"), ("big", "felines")
    #     }
    # }

    def generate(self, num_sentences: int, side: str = 'both', single_synonyms: Optional[List[int]] = None) -> List[str]:
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
        return sentences[:num_sentences]

    def generate_all_syns(self, side: str = 'both', single_synonyms: Optional[List[int]] = None) -> List[str]:
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

            for i in range(max(len(sublist) for sublist in adjusted_lists)):
                sentence = " ".join([s[i % len(s)] for s in adjusted_lists])
                if sentence not in sentences:
                    sentences.append(sentence)
            return sentences

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
        return sentences


if __name__ == "__main__":
    synonyms = Synonyms()
    # sentences = synonyms.generate(500, side='both', single_synonyms=[0, 4])
    sentences = synonyms.generate(50000, side='both')
    # sentences = synonyms.generate_all_syns(side='both')
    for sentence in sentences:
        print(sentence.capitalize(), end='. ')
    print()
    print(len(sentences))
