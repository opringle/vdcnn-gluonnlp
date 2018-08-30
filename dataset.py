import numpy as np
from mxnet import gluon
import random


class UtteranceDataset(gluon.data.ArrayDataset):
    """
    preprocesses text
    """
    def __init__(self, data, labels, alphabet, feature_len):
        super().__init__(data, labels)
        self.alphabet = alphabet
        self.feature_len = feature_len
        self.char_to_index = {letter: index for index, letter in enumerate(alphabet)}
        self.label_to_index = {label: index for index, label in enumerate(set(labels))}

    def encode(self, text):
        encoded = np.zeros([self.feature_len], dtype='float32')
        i = 0
        for letter in text:
            if i >= self.feature_len:
                break
            encoded[i] = self.char_to_index.get(letter, -1)
            i += 1
        return encoded

    def __getitem__(self, idx):
        return self.encode(self._data[0][idx]), self.label_to_index[self._data[1][idx]]


if __name__ == "__main__":
    """
    Run unit-test
    """
    alph = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
    alph_idx = {letter: index for index, letter in enumerate(alph)}
    labels = ['Business', 'Sci/Tech', 'Sports', 'World']
    label_idx = {label: index for index, label in enumerate(labels)}
    max_chars = 1014

    utterances = [''.join([random.choice(alph) for _ in range(random.randint(1, 100))]) for i in range(100)]
    labels = [random.choice(labels) for _ in range(100)]

    dataset = UtteranceDataset(data=utterances, labels=labels, alphabet=alph, feature_len=max_chars)

    assert len(dataset) == 100
    for i in range(len(dataset)):
        d, l = dataset[i]
        assert d.shape == (max_chars,)
        assert type(l) is int
    print("Unit-test success!")
