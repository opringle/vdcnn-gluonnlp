from mxnet import gluon
import random


class UtteranceDataset(gluon.data.ArrayDataset):
    def __init__(self, data, labels, alphabet):
        """
        :param data: list of list of text strings
        :param labels: list of integer labels
        :param alphabet: list of characters
        :param feature_len: max characters per text string
        """
        super().__init__(data, labels)
        self.alphabet = alphabet
        self.char_to_index = {letter: index for index, letter in enumerate(alphabet)}
        self.label_to_index = {label: index for index, label in enumerate(set(labels))}

    def encode(self, text):
        """
        index character data
        :param text: string to index
        :return: list of int
        """
        return [self.char_to_index.get(letter, -1) for letter in text]

    def __getitem__(self, idx):
        return self.encode(self._data[0][idx]), self.label_to_index[self._data[1][idx]]


if __name__ == "__main__":
    """
    Run unit-test
    """
    alph = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
    labels = ['Business', 'Sci/Tech', 'Sports', 'World']
    label_idx = {label: index for index, label in enumerate(labels)}

    utterances = [''.join([random.choice(alph) for _ in range(random.randint(1, 100))]) for i in range(100)]
    labels = [random.choice(labels) for _ in range(100)]

    dataset = UtteranceDataset(data=utterances, labels=labels, alphabet=alph)

    assert len(dataset) == 100
    for i in range(len(dataset)):
        d, l = dataset[i]
        assert type(l) is int
    print("Unit-test success!")
