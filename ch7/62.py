# 62

from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

for i, (word, sim) in enumerate(wv.most_similar('United_States')):
  print(f'{i+1}: {word}, {sim}')