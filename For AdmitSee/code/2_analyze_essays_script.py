import pandas as pd
import numpy as np
import essay_analysis as ea
import cPickle as pickle
import json

df = pd.read_csv('../data/cleaned_data.csv', low_memory=False)
df2 = df[df['essay_final'].notnull()].copy()
essays = df2['essay_final'].values.copy()
essays_idx = df2['id'].values
X = essays.copy()

# Remove stop words and stem/lemmatize
ae = ea.AnalyzeEssays()
ae.stopWordsAndStem(X)

# Load the TFIDF, Transform
with open('../data/vectorizer.pkl', 'r') as f:
    vec = pickle.load(f)
mat = vec.transform(X)

# Load the NMF, Transform
with open('../data/nmf.pkl', 'r') as f:
    nmf = pickle.load(f)
W = nmf.transform(mat)

# Pickle the TFIDF mat
with open('../data/tfidf_mat.pkl', 'w') as f:
    pickle.dump(mat, f)

###################

# Create df for just essays and topics
essays_and_topics = np.hstack((essays_idx.reshape(-1,1), (essays.reshape(-1,1))))
essays_and_topics = np.hstack((essays_and_topics, W))
topics = ['family','music','culture','sport','personal','science','career']
df_et_cols = ['id','content'] + topics
df_et = pd.DataFrame(essays_and_topics, columns=df_et_cols)

# Export
df_et.to_csv('../data/essays_and_topics.csv')

print 'Finished analyzing all essays.\nNew file created/updated: essays_and_topics.csv'
