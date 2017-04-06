import gensim
from gensim.models.keyedvectors import KeyedVectors
# numpy
import numpy as np
from sklearn.utils import shuffle
from file_io import FileIO
print('loading word2vec model...')
# load google pretrained word2vec model
word_vectors = KeyedVectors.load_word2vec_format(
    './word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)

print('loading text...')
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import operator, functools

labels = []
doc_vectors = []
with open('./datasets/all_data_set.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    lines = [line.split('|sep|') for line in lines]
    print(len(lines))
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # English stop words
    stops = set(stopwords.words("english"))
    for line in lines:
        label = line[0]
        text = line[1]
        # Remove HTML
        text = BeautifulSoup(text).get_text()
        # Remove non-letters
        text = re.sub("[^a-zA-Z]", " ", text)
        # Convert words to lower case
        words = text.lower().split()
        words = [w for w in words if not w in stops]
        if len(words) <= 3:
            continue
        vectors = []
        for word in words:
            try:
                vectors.append(word_vectors[word])
            except Exception as e:
                # print(e)
                pass

        average_vector = functools.reduce(np.add, vectors)
        average_vector = average_vector / 300
        labels.append(label)
        doc_vectors.append(average_vector)

print('shuffle vectors')
labels = np.array(labels)
doc_vectors = np.array(doc_vectors)
print(labels.shape)
print(doc_vectors.shape)
# doc_vectors, labels = shuffle(doc_vectors, labels, random_state=0)
print('write output')
lines = []
for i in range(100000):
    vector = doc_vectors[i]
    vector = ["%.4f" % item for item in vector.tolist()]
    label = labels[i]
    lines.append(label + '|sep|' + ','.join(vector))
FileIO.write_lines_to_file('./datasets/word2vec_ag12bbc.txt', lines)

