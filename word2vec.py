import gensim
from gensim.models.keyedvectors import KeyedVectors
# numpy
import numpy as np

print('loading word2vec model...')
# load google pretrained word2vec model
word_vectors = KeyedVectors.load_word2vec_format(
    './word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)

print('loading text...')
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data

sentences = []
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
        text = ' '.join(words)
        if len(text.split()) <= 3:
            continue
        vectors = []
        for word in text:
            if text in word_vectors.vocab:
                vectors.append(word_vectors[text])

        print(len(vectors))
        average_vector = np.sum(vectors) / 300
        print(len(average_vector))
        new_line = label + '|sep|' + text
        break
        sentences.append(new_line)
