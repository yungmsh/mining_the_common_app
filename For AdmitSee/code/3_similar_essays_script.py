import pandas as pd
import numpy as np
import cPickle as pickle
from scipy.spatial.distance import euclidean
import re
import json
import essay_analysis as ea

with open('../data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)
with open('../data/tfidf_mat.pkl') as f:
    tfidf_mat = pickle.load(f)
with open('../data/nmf.pkl') as f:
    nmf = pickle.load(f)

def processEssay(essay, similarity='word'):
    '''
    This function analyzes the essay and outputs a list of two items. The first is a list tuples of the following form: (id, essay). The second is a dict of the topic distribution.

    Args:
        essay (str): essay text
        similarity (str): 'word' or 'topic'
    Returns:
        output (list): contains two items: a list of tuples of the form (similar IDs, similar essays) and a dict of topic distribution
    '''
    mat = vectorizer.transform([essay])
    mat_nmf = nmf.transform(mat)
    mat_nmf_wt = mat_nmf[0] / mat_nmf[0].sum() * 100
    topic1 = mat_nmf_wt[0].round(1)
    topic2 = mat_nmf_wt[1].round(1)
    topic3 = mat_nmf_wt[2].round(1)
    topic4 = mat_nmf_wt[3].round(1)
    topic5 = mat_nmf_wt[4].round(1)
    topic6 = mat_nmf_wt[5].round(1)
    topic7 = mat_nmf_wt[6].round(1)
    topics = [topic1, topic2, topic3, topic4, topic5, topic6, topic7]

    # Load in database of essays and topics
    df_essay = pd.read_csv('../data/essays_and_topics.csv')
    essays = df_essay['content'].values
    essays_id = df_essay['id'].values
    essays_dict = {essay:i for essay,i in zip(essays, essays_id)}
    topic_names = ['family', 'music', 'culture', 'sport', 'personal', 'science', 'career']
    topic_mat = df_essay.loc[:,topic_names].values

    # Get essays based on NMF euclidean distance or TFIDF cosine similarity
    tm = ea.TopicModeling()
    if similarity == 'word':
        similar_essays = tm.similarEssaysTfidf(essay, essays, tfidf_mat, vectorizer, n=50)
    elif similarity == 'topic':
        similar_essays = tm.similarEssaysNMF(essay, essays, topic_mat, vectorizer, nmf, n=50)

    similar_ids = [essays_dict[essay] for essay in similar_essays]
    similar_tuples = zip(similar_ids, similar_essays)
    topic_distr = {t:val for t,val in zip(topic_names, topics)}

    return [similar_tuples, topic_distr]

if __name__ == '__main__':
    filename = '../data/sample_essay.txt'

    with open(filename, 'r') as f:
        essay = f.read()
    similar_tuples, topic_distr = processEssay(essay, similarity='word')

    sub_json = [{'id':userid, 'essay':content} for userid,content in similar_tuples]
    json_output = {'similar essays': sub_json, 'topic distribution': topic_distr}

    with open('../data/similar_essays.json', 'w') as j:
        json.dump(json_output, j)

    print 'Finished analyzing your essay.\nNew file created/updated: similar_essays.json'
