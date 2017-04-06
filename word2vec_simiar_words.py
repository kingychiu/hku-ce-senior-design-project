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
with open('./datasets/word2vec_ag12bbc.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    lines = [line.split('|sep|') for line in lines]
    print(len(lines))
    for line in lines:
        label = line[0]
        vector = line[1]
        bag_of_words = {}
        # get similar word of the word
        try:
            simiar_words = word_vectors.similar_by_vector(vector, topn=10)
            for w in simiar_words:
                if w[0] in bag_of_words.keys():
                    bag_of_words[w[0]] += w[1]
                else:
                    bag_of_words[w[0]] = w[1]
        except:
            pass

        print(bag_of_words)
        related_words.append(sorted(bag_of_words.items(), key=operator.itemgetter(1), reverse=True)[10:])
        print(len(related_words))
        break
        # print(text)
        # for s in sorted(bag_of_words.items(), key=operator.itemgetter(1), reverse=True):
        #     if s[1] != 0:
        #         print('\t', s[0], s[1])


print('write output')
lines = []
for i in range(100000):
    words = related_words[i]
    label = labels[i]
    lines.append(label + '|sep|' + ','.join(words))
FileIO.write_lines_to_file('./datasets/all_related_words.txt', lines)
