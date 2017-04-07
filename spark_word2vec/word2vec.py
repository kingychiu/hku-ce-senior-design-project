from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec,Word2VecModel

conf = SparkConf().setAppName('Word2Vec').setMaster('spark://cep16001s1:7077').set('spark.eventLog.enabled', True)
sc = SparkContext(conf=conf)

lines = []
with open('dataset-after-clean.txt', 'r', encoding='utf8') as f:
  lines = f.readlines()
  lines = [l.replace('\n','') for l in lines]


doc = sc.parallelize(lines).map(lambda line: line.split(" "))
model = Word2Vec().setVectorSize(300).setMinCount(30).fit(doc)

model.save(sc, 'ag_word2vec_n_300.model')
sc.stop()
# print("<==RESULT")
# synonyms = model.findSynonyms('computer', 5)
# print ('RESULT==>')