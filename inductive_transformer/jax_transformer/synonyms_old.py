from dataclasses import dataclass


@dataclass
class Synonyms:
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
    valid_right_zeroth_words = [
        "angry",
        "hateful",
        "mean",
        "nasty",
        "unpleasant",
        "vicious",
        "violent",
        "wicked",
    ]
    valid_left_first_words = [
        "dogs",
        "canines",
    ]
    valid_right_first_words = [
        "birds",
        "avians",
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

    valid_right_fourth_words = [
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

    valid_right_fifth_words = [
        "worms",
        "earthworms"
    ]



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
            }
            | {
                (a, b)
                for a in self.valid_right_zeroth_words
                for b in self.valid_right_first_words
            },
            (1, 3): {
                (a, b)
                for a in self.valid_left_first_words
                for b in self.valid_left_third_words
            }
            | {
                (a, b)
                for a in self.valid_right_first_words
                for b in self.valid_right_third_words
            },
            (2, 3): {
                (a, b)
                for a in self.valid_left_second_words
                for b in self.valid_left_third_words
            }
            | {
                (a, b)
                for a in self.valid_right_second_words
                for b in self.valid_right_third_words
            },
            (3, 5): {
                (a, b)
                for a in self.valid_left_third_words
                for b in self.valid_left_fifth_words
            }
            | {
                (a, b)
                for a in self.valid_right_third_words
                for b in self.valid_right_fifth_words
            },
            (4, 5): {
                (a, b)
                for a in self.valid_left_fourth_words
                for b in self.valid_left_fifth_words
            }
            | {
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
