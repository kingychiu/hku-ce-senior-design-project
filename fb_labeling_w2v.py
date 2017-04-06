# numpy
import numpy as np

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from file_io import FileIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import operator
import gensim
from gensim.models.keyedvectors import KeyedVectors

# load knn
print('loading knn')

stat = {}

with open('./datasets/word2vec_ag12bbc.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    print(len(lines))
    labels = []
    features = []
    for line in lines:
        label = line.split('|sep|')[0]
        if label in stat.keys() and stat[label] < 5000:
            stat[label] += 1
            labels.append(label)
            features.append(line.split('|sep|')[1].split(','))
        elif label not in stat.keys():
            print(label)
            stat[label] = 1
            labels.append(label)
            features.append(line.split('|sep|')[1].split(','))
        else:
            pass
    print(features)
    break
    f.close()

print(stat)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,
                                                    random_state=42)
classes = sorted(list(set(y_train)))
print(classes)
print('train', len(x_train))
print('test', len(x_test))
del labels
del features

neigh = KNeighborsClassifier(n_neighbors=50)
neigh.fit(x_train, y_train)


print('loading model')
# load google pretrained word2vec model
word_vectors = KeyedVectors.load_word2vec_format(
    './word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)


def get_deep_features(string):
    pass

word_vectors = []
with open('./datasets/word2vec_ag12bbc.txt', 'r', encoding='utf8') as original_data:
    lines = original_data.readlines()
    word_vectors = [l.split('|sep|')[1].replace('\n', '').split(',') for l in lines]
    word_vectors = [map(float, v) for v in word_vectors]

def get_labels(string):
    sample = get_deep_features(string)
    distances, neighbors = neigh.kneighbors(sample)
    return neighbors[0]




# print('ready')
with open('./fb_posts/Cristiano.txt', 'r', encoding='utf8') as fb_posts:
    lines = fb_posts.readlines()
    count = 0
    for line in lines:
        neighbors = get_labels(line)
        for n in neighbors:
            print(n, word_vectors[n])
            print(trained_word2vec.similar_by_vector(np.array(word_vectors[n]), topn = 5))
        count += 1
        print(count)
        break
