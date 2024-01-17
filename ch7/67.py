# 67

import pycountry
from sklearn.cluster import KMeans
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

print(len(countries))
print(len(country_vec))

model = KMeans(n_clusters=5, random_state=0).fit(country_vec)
res = defaultdict(list)
for i, label in enumerate(model.labels_):
  res[label].append(countries[i])

for i in range(5):
  print(f'---{i}---')
  print(res[i])