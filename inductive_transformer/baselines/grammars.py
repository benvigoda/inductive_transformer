class BigCatSmallDog:
    small_synonyms = [
        "small",
        "tiny",
        "little",
        "micro",
        "pico",
        "femto",
        "diminimus",
        "itty",
        "teenyweeny",
    ]

    big_synonyms = [
        "big",
        "giant",
        "gargantuan",
        "enormous",
        "huge",
        "large",
        "extralarge",
    ]

    cat_synonyms = ["cat", "feline"]
    dog_synonyms = ["dog", "canine"]

    def is_valid(self, sentence):
        words = sentence.split()
        assert len(words) == 2
        if words[0] in self.big_synonyms and words[1] in self.cat_synonyms:
            return True
        if words[0] in self.small_synonyms and words[1] in self.dog_synonyms:
            return True
        return False

    def all_valid_sentences(self):
        sentences = []
        for big_synonym in self.big_synonyms:
            for cat_synonym in self.cat_synonyms:
                sentences.append((big_synonym, cat_synonym))
        for small_synonym in self.small_synonyms:
            for dog_synonym in self.dog_synonyms:
                sentences.append((small_synonym, dog_synonym))
        return sentences
