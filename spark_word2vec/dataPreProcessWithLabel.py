from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data

# with open('dataset-after-clean-with-label.txt','r',encoding='utf8') as f1:
#   l = f1.readlines()
#   print(l[0])
with open('../dataset/ag_dataset_50000_each.txt', 'r', encoding='utf8') as f:
  lines = f.readlines()
  lines = [line.split('\\C') for line in lines]
  print(len(lines))
  # Load the punkt tokenizer
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  # English stop words
  stops = set(stopwords.words("english"))

  # for training word2vec
  new_lines = []
  for line in lines:
    label = line[0]
    text = line[1]
    # Remove HTML
    text = BeautifulSoup(text).get_text()
    # Remove non-letters
    text  = re.sub("[^a-zA-Z]"," ", text)
    # Convert words to lower case
    words = text.lower().split()
    words = [w for w in words if not w in stops]
    text = ' '.join(words)
    if len(text.split())<=3:
      continue
    new_line = label +'\\C'+text
    new_lines.append(new_line)

  with open('dataset-after-clean-with-label-50000-each.txt','w',encoding='utf8') as f1:
    print(len(new_lines))
    f1.write('\n'.join(new_lines))


    