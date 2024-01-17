# 68

import pycountry
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from gensim.models import KeyedVectors
import numpy as np
from collections import defaultdict

wv = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

countries = []
country_vec = []
for country in list(pycountry.countries):
  name = country.name
  if name in wv:
    countries.append(name)
    country_vec.append(wv[name])

linkage_result = linkage(country_vec, method='ward')

plt.figure(figsize=(16, 9))
dendrogram(linkage_result, labels=countries)
plt.savefig('graph68.png')