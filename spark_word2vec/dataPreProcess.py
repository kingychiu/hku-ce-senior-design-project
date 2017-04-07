from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data

with open('../dataset/ag_dataset.txt', 'r', encoding='utf8') as f:
  lines = f.readlines()
  lines = [line.split('\\C')[1] for line in lines]
  print(len(lines))
  # Load the punkt tokenizer
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  # English stop words
  stops = set(stopwords.words("english"))

  # for training word2vec
  new_lines = []
  for line in lines:
    sentences = []
    # break each document into sentances
    raw_sentences = tokenizer.tokenize(line.strip())
    for sentence in raw_sentences:
      # Remove HTML
      sentence = BeautifulSoup(sentence).get_text()
      # Remove non-letters
      sentence  = re.sub("[^a-zA-Z]"," ", sentence)
      # Convert words to lower case
      words = sentence.lower().split()
      words = [w for w in words if not w in stops]
      sentence = ' '.join(words)
      if len(sentence.split())<=3:
        continue
      sentences.append(sentence)
    new_lines += sentences
    
  with open('dataset-after-clean-for-train-word2vec.txt','w',encoding='utf8') as f1:
    print(len(new_lines))
    f1.write('\n'.join(new_lines))


    