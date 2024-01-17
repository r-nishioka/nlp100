# 64

from gensim.models import KeyedVectors
from collections import defaultdict
import pandas as pd


def calc_most_sim(wv, p_list, n_list):
  word, sim = wv.most_similar(positive=p_list, negative=n_list, topn=1)[0]
  return word, sim
  

wv = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
questions = defaultdict()
values = defaultdict(list)
with open('data/questions-words.txt') as f:
  for l in f.readlines():
    if l[0] == ':':
      if values:
        questions[k] = pd.DataFrame(values)
      values = defaultdict(list)
      k = l[2:].strip()
    else:
      w1, w2, w3, w4 = l.strip().split(' ')
      values['word1'].append(w1)
      values['word2'].append(w2)
      values['word3'].append(w3)
      values['word4'].append(w4)

with open('ans64.txt', mode='w') as f:
  for k, df in questions.items():
    for i in range(df.shape[0]):
      w1, w2, w3, w4 = df.iloc[i, :]
      word, sim = calc_most_sim(wv, p_list=[w2, w3], n_list=[w1])
      f.write(f'{k} {w1} {w2} {w3} {w4} {word} {sim}\n')
