# 63

from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

for i, (word, sim) in enumerate(wv.most_similar(positive=['Spain','Athens'], negative=['Madrid'])):
  print(f'{i+1}: {word}, {sim}')