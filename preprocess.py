# numpy
import numpy as np
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# other classes
from file_io import FileIO


class PreProcess:
    def __init__(self, file_path):
        self.file_path = file_path

    def run(self):
        lines = FileIO.read_file_to_lines(self.file_path)
        # spiting labels and sentences
        sentences = [l.split('\\C')[1].split(',')[0] for l in lines]
        print('## First sentence')
        print(sentences[0])
        print()
        sentences_in_ascii = [self.sentence_to_ascii_list(sentence) for sentence in sentences]
        print('## First sentence in ascii codes')
        print(sentences_in_ascii[0])
        print()
        x = np.asarray(sentences_in_ascii)
        print('# Whole Data Set', x.shape)
        print()
        y = np.asarray([l.split('\\C')[0] for l in lines])
        # classes
        classes = sorted(list(set(y)))
        print('Labels', classes)
        print()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        print('# Training Data', x_train.shape)
        print('# Testing Data', x_test.shape)
        return x_train, x_test, y_train, y_test, len(classes)

    def sentence_to_ascii_list(self, s):
        return [ord(char) for char in list(s)]

    def train_test_split(self, x, y):
        # shuffle
        x, y = shuffle(x, y, random_state=0)
        return train_test_split(x, y, test_size=0.3, random_state=42)