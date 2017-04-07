from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, Word2VecModel
from pyspark.mllib.common import _java2py
from pyspark.mllib.linalg import Vectors
# local spark
print('LOADING MODEL')
conf = SparkConf().setAppName('ReadModel').setMaster('spark://cep16001s1:7077')
sc = SparkContext(conf=conf)

n = 1000
model = Word2VecModel.load(sc, 'ag_word2vec_n_'+str(n)+'.model')
print('LOADING DATASET')
# result = model.transform("computer")
new_lines = []
with open('dataset-after-clean-with-label-10000-each.txt','r',encoding='utf8') as f:
    lines = f.readlines()
    total = len(lines)
    count = 0
    for line in lines:
      count += 1
      if count % 1000 == 0:
        print(count, total)
      label = line.split('\\C')[0]
      text = line.split('\\C')[1]
      words = text.split()
      vecs = []
      for word in words:
        try:
          vec = model.transform(word)
          vecs.append(vec)
        except:
          pass
      # print(text)
      # print('A word vector:')
      t = vecs[0]
      ty = type(vecs[0])
      # print(ty)
      # print(len(t))
      avg_vec = Vectors.dense([0] * len(t))
      for vec in vecs:
        avg_vec += vec
      avg_vec = avg_vec / len(t)
      # print('avg. vector:')
      # print(type(avg_vec))
      # print(len(avg_vec))
      # print('avg vector synonyms:')
      # synonyms = list(model.findSynonyms(avg_vec, 10))
      # for word, sim in synonyms:
      #   print(word, sim)
      avg_vec_str = ','.join([str(val) for val in avg_vec])
      new_line = label+'\\C'+avg_vec_str
      # print(new_line)
      new_lines.append(new_line)
print('start writing')
with open('ag_word2vec_n_'+str(n)+'.dataset','w',encoding='utf8') as f1:
    f1.write('\n'.join(new_lines))



    