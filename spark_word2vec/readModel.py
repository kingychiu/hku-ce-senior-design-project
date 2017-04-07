from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, Word2VecModel
from pyspark.mllib.common import _java2py
# local spark
conf = SparkConf().setAppName('ReadModel').setMaster('local')
sc = SparkContext(conf=conf)

model = Word2VecModel.load(sc, 'ag_word2vec_n_300.model')

# sysn_lst = list(synonyms)
result = model.transform("computer")
vector = model.getVectors()
sc.stop()


def printNumberFeatures():
    print('# of features')
    print(len(result))
def printSimilarWords(input):
    synonyms = list(model.findSynonyms(input, 10))
    print ('<==RESULT')
    for word, sim in synonyms:
        print(word, sim)
    print ('RESULT==>')
def getVector(word):
    v = model.transform(word)
    print(v)
