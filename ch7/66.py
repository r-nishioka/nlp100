# 66

from scipy.stats import spearmanr
from gensim.models import KeyedVectors
import pandas as pd

wv = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

df_human = pd.read_csv('./data/wordsim353/combined.csv')
df_wv = pd.DataFrame(columns=['Word 1', 'Word 2', 'sim_score'])

df_wv['Word 1'] = df_human['Word 1']
df_wv['Word 2'] = df_human['Word 2']
for i in df_wv.index:
	df_wv.at[i, 'sim_score'] = wv.similarity(df_wv.at[i, 'Word 1'], df_wv.at[i, 'Word 2'])

corr, p_value = spearmanr(df_human['Human (mean)'].rank(ascending=False), df_wv['sim_score'].rank(ascending=False))
print(corr)