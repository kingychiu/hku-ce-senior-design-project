# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# other classes
from file_io import FileIO
import re
import numpy as np


class PreProcess:
    def __init__(self, file_path):
        self.file_path = file_path

    def run_look_up(self):
        lines = FileIO.read_file_to_lines(self.file_path)
        # spiting labels and sentences
        sentences = [l.split('|sep|')[1] for l in lines]
        print('## First sentence')
        print(sentences[0])
        print('# char', len(sentences[0]))

        avg_char = 0
        for s in sentences:
            avg_char += len(s)
        print('avg char:', avg_char / len(sentences))
        sentences_in_ascii = [self.sentence_to_ascii_list_look_up(sentence) for sentence in
                              sentences]
        print('## First sentence in ascii codes')
        print(sentences_in_ascii[0])
        print()

        sentences_in_ascii = self.fixing_dimension(sentences_in_ascii)
        sentences_in_vector = self.generate_look_up_vector(sentences_in_ascii)
        labels = [l.split('|sep|')[0] for l in lines]
        return labels, sentences_in_vector

    def sentence_to_ascii_list_look_up(self, s):
        return [self.char2lookup(char) for char in list(s)]

    def sentence_to_ascii_list(self, s):
        return [ord(char) for char in list(s)]

    def char2lookup(self, char):
        if ord(char) >= 32 and ord(char) <= 127:
            return ord(char) - 32
        else:
            return 95

    def train_test_split(self, x, y):
        # shuffle
        x, y = shuffle(x, y, random_state=0)
        return train_test_split(x, y, test_size=0.3, random_state=42)

    def fixing_dimension(self, data):
        fix_size = 100
        avg = 0
        max_char = len(data[0])
        min_char = len(data[0])
        for i in range(0, len(data)):
            if len(data[i]) > max_char:
                max_char = len(data[i])
            if len(data[i]) < min_char:
                min_char = len(data[i])
            avg += len(data[i])
        avg /= len(data)
        print('avg char', avg)
        print('max char', max_char)
        print('min char', min_char)
        for i in range(0, len(data)):
            if len(data[i]) >= fix_size:
                data[i] = data[i][:fix_size]
            else:
                diff = fix_size - len(data[i])
                data[i] += (diff * [26])
        print('Fixing all sentences to ', fix_size, ' char')
        return data

    def generate_look_up_vector(self, data):
        # create one hot vector
        look_up_tensor = []
        for d in data:
            look_up_matrix = []
            for char in d:
                binary_look_up = '{0:07b}'.format(char)
                if (len(binary_look_up) != 7):
                    print(len(binary_look_up))
                look_up_matrix.append(binary_look_up)
            look_up_tensor.append(look_up_matrix)
        return look_up_tensor
