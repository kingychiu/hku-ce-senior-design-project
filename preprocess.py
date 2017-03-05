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

        sentences_in_ascii = self.fixing_dimension(sentences_in_ascii)
        x = np.asarray(sentences_in_ascii)

        print('# Whole Data Set', x.shape)
        print()
        y = [l.split('\\C')[0] for l in lines]
        # classes
        classes = sorted(list(set(y)))
        y = np.asarray([classes.index(item) for item in y])
        print('Labels', classes)
        print()
        x_train, x_test, y_train, y_test = self.train_test_split(x, y)
        print('# Training Data', x_train.shape, y_train.shape)
        print('# Testing Data', x_test.shape, y_test.shape)
        return x_train, x_test, y_train, y_test, len(classes)

    def sentence_to_ascii_list(self, s):
        return [ord(char) for char in list(s)]

    def train_test_split(self, x, y):
        # shuffle
        x, y = shuffle(x, y, random_state=0)
        return train_test_split(x, y, test_size=0.3, random_state=42)

    def fixing_dimension(self, data):
        avg_len = 0
        for d in data:
            avg_len += len(d)
        avg_len //= len(data)
        for i in range(0, len(data)):
            if len(data[i]) >= avg_len:
                data[i] = data[i][:avg_len]
            else:
                diff = avg_len - len(data[i])
                data[i] += (diff * [0])
        print('Fixing all sentences to ', avg_len, ' char')
        return data
