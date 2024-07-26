from dataclasses import dataclass

@dataclass
class Synonyms:

    valid_first_words = [
        "small", "little", "tiny", "micro", "mini",
    ]
    valid_second_words = [
        "dogs", "canines",
    ]
    valid_third_left_words = [
        "often", "usually", "commonly", "frequently",
    ]
    valid_third_right_words = [
        "sometimes", "occasionally", "rarely", "never",
    ]
    valid_fourth_left_words = [
        "fear", "avoid",
    ]
    valid_fourth_right_words = [
        "chase", "intimidate", "eat",
    ]
    valid_fifth_words = [
        "large", "giant", "huge", "humongous", "enormous", "big",
    ]
    valid_sixth_words = [
        "cats", "felines",
    ]

    valid_first_pairs = {
        (1, 2): {
            ("small", "dogs"), ("small", "canines"),
            ("little", "dogs"), ("little", "canines"),
            ("tiny", "dogs"), ("tiny", "canines"),
            ("micro", "dogs"), ("micro", "canines"),
            ("mini", "dogs"), ("mini", "canines"),
        }
    }
    valid_first_pairs = {
        (1, 2): {
            ("small", "dogs"), ("small", "canines"),
            ("little", "dogs"), ("little", "canines"),
            ("tiny", "dogs"), ("tiny", "canines"),
            ("micro", "dogs"), ("micro", "canines"),
            ("mini", "dogs"), ("mini", "canines"),
            ("extralarge", "cats"), ("extralarge", "felines"),
            ("gargantuan", "cats"), ("gargantuan", "felines"),
            ("large", "cats"), ("large", "felines"),
            ("giant", "cats"), ("giant", "felines"),
            ("huge", "cats"), ("huge", "felines"),
            ("humongous", "cats"), ("humongous", "felines"),
            ("enormous", "cats"), ("enormous", "felines"),
            ("big", "cats"), ("big", "felines"),
            ("pico", "dogs"), ("pico", "canines"),
            ("femto", "dogs"), ("femto", "canines"),
            ("diminimus", "dogs"), ("diminimus", "canines"),
            ("itty", "dogs"), ("itty", "canines"),
            ("teenyweeny", "dogs"), ("teenyweeny", "canines"),
        }
    }

    valid_middle_pairs = {
        (3, 4): {
            ("often", "fear"), ("often", "avoid"),
            ("usually", "fear"), ("usually", "avoid"),
            ("commonly", "fear"), ("commonly", "avoid"),
            ("frequently", "fear"), ("frequently", "avoid"),
            ("sometimes", "chase"), ("sometimes", "intimidate"), ("sometimes", "eat"),
            ("occasionally", "chase"), ("occasionally", "intimidate"), ("occasionally", "eat"),
            ("rarely", "fear"), ("rarely", "avoid"),
            ("never", "fear"), ("never", "avoid")
        }
    }

    valid_last_pairs = {
        (5, 6): {
            ("large", "cats"), ("large", "felines"),
            ("giant", "cats"), ("giant", "felines"),
            ("huge", "cats"), ("huge", "felines"),
            ("humongous", "cats"), ("humongous", "felines"),
            ("enormous", "cats"), ("enormous", "felines"),
            ("big", "cats"), ("big", "felines")
        }
    }
