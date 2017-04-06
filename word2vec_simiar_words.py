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
related_words = []
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
        bag_of_words = {}
        for word in words:
            # get similar word of the word
            try:
                simiar_words = word_vectors.similar_by_word(word, topn=10)
                for w in simiar_words:
                    if w[0] in bag_of_words.keys():
                        bag_of_words[w[0]] += w[1]
                    else:
                        bag_of_words[w[0]] = w[1]
            except:
                pass
        related_words.append(sorted(bag_of_words.items(), key=operator.itemgetter(1), reverse=True)[10:])
        # print(text)
        # for s in sorted(bag_of_words.items(), key=operator.itemgetter(1), reverse=True):
        #     if s[1] != 0:
        #         print('\t', s[0], s[1])



# labels = np.array(labels)
# doc_vectors = np.array(doc_vectors)
# print(labels.shape)
# print(doc_vectors.shape)
# # doc_vectors, labels = shuffle(doc_vectors, labels, random_state=0)
print('write output')
lines = []
for i in range(100000):
    words = related_words[i]
    label = labels[i]
    lines.append(label + '|sep|' + ','.join(words))
FileIO.write_lines_to_file('./datasets/all_related_words.txt', lines)
