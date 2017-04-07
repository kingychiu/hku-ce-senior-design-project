from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, Word2VecModel
from pyspark.mllib.common import _java2py
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.sql.functions import col

# local spark
print('LOADING MODEL')
n = 100
conf = SparkConf().setAppName('AverageVector').setMaster('spark://cep16001s1:7077').set("spark.driver.maxResultSize", "2g").set('spark.eventLog.enabled', True)

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

lookup = sqlContext.read.parquet('ag_word2vec_n_'+str(n)+'.model/data').alias("lookup")
lookup.printSchema()
lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())

def line_handler(line):
  label = line.split('\\C')[0]
  text = line.split('\\C')[1]
  words = text.split()
  vecs = []
  for word in words:
    vec = Vectors.dense(lookup_bd.value.get(word))
    vecs.append(vec)
  # remove unvalid Vectors
  vecs = [v for v in vecs if v.size == n]
  num_words = len(vecs)
  
  avg_vec = Vectors.dense([0] * n)
  for vec in vecs:
    avg_vec += vec
  avg_vec = avg_vec / num_words
  avg_vec_str = ','.join([str(val) for val in avg_vec])
  return label + '\\C' + avg_vec_str



with open('dataset-after-clean-with-label-10000-each.txt','r',encoding='utf8') as f:
  lines = f.readlines()
  x = sc.parallelize(lines)
  y = x.map(line_handler)
  y = y.collect()
  print('start writing')
  with open('ag_word2vec_n_'+str(n)+'.dataset','w',encoding='utf8') as f1:
      print(len(y))
      for line in y:
        f1.write(line+'\n')
      print('done writing')

sc.stop()

    