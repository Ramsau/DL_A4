from collections import Counter
import re

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class Vocabulary:
    def __init__(self, min_freq=1, specials=['<pad>', '<unk>']):
        self.min_freq = min_freq
        self.specials = specials
        self.stoi = {}
        self.itos = []
        for token in specials:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = tokenize(text)
            counter.update(tokens)
        for token, freq in counter.items():
            if freq >= self.min_freq:
                self.add_token(token)

    def numericalize(self, text):
        tokens = tokenize(text)
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]