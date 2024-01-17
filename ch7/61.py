# 61

from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
print(wv.similarity('United_States', 'U.S.'))