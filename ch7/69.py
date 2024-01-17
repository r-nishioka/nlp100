# 69

import pycountry
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

model = TSNE(n_components=2, random_state=0)
res = model.fit_transform(country_vec)
x, y = [list(i) for i in zip(*res)]

plt.figure(figsize=(10, 10))
plt.xlim([min(x)-10, max(x)+10])
plt.ylim([min(y)-10, max(y)+10])
for country, x, y in zip(countries, x, y):
  plt.text(x, y, country)
plt.savefig('graph69.png')