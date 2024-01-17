# 65

from gensim.models import KeyedVectors
from collections import defaultdict
import pandas as pd  

questions = defaultdict()
values = defaultdict(lambda: defaultdict(list))

with open('ans64.txt', mode='r') as f:
  for l in f.readlines():
    k, w1, w2, w3, w4, word, sim = l.strip().split(' ')
    if k[:4] == 'gram':
      category = 'syntactic analogy'
    else:
      category = 'semantic analogy'
    values[category]['word1'].append(w1)
    values[category]['word2'].append(w2)
    values[category]['word3'].append(w3)
    values[category]['word4'].append(w4)
    values[category]['most_sim_word'].append(word)

for category in ['semantic analogy', 'syntactic analogy']:
  questions[category] = pd.DataFrame(values[category])

for k, df in questions.items():
  df['correct'] = 0
  for i in range(df.shape[0]):
    if df.at[i, 'word4'] == df.at[i, 'most_sim_word']:
      df.at[i, 'correct'] = 1
  
  vc = df['correct'].value_counts()
  acc = vc[1] / vc.sum()
  
  print(f'accuracy of {k}: {acc}')