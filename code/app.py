import pandas as pd
import numpy as np
import cPickle as pickle
from flask import Flask, request, render_template, jsonify
from scipy.spatial.distance import euclidean
from flask_bootstrap import Bootstrap
import re
from summa import textrank
from nltk.tokenize.punkt import PunktSentenceTokenizer
import essay_analysis as ea

app = Flask(__name__)
Bootstrap(app)

with open('../data/app/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)
with open('../data/app/tfidf_mat.pkl') as f:
    tfidf_mat = pickle.load(f)
with open('../data/app/nmf.pkl') as f:
    nmf = pickle.load(f)
with open('../data/app/model.pkl') as f:
    my_model = pickle.load(f)

# home page
@app.route('/')
def index():
	return render_template('index.html', title = 'Welcome')

@app.route('/model', methods = ['GET', 'POST'])
def model():
    male = request.form.get('male')
    sat = request.form.get('sat')
    sat_times_taken = request.form.get('sat_times_taken')
    gpa = request.form.get('gpa')
    asian = request.form.get('asian')
    black = request.form.get('black')
    hispanic = request.form.get('hispanic')
    white = request.form.get('white')
    pacific = request.form.get('pacific')
    nativeam = request.form.get('nativeam')
    ecc = request.form.get('ecc')
    return render_template('model.html', sat = sat, sat_times_taken = sat_times_taken, gpa = gpa, male=male, asian=asian, black=black, hispanic=hispanic, white=white, pacific=pacific, nativeam=nativeam, ecc=ecc)

@app.route('/model_results', methods = ['POST'])
def model_results():
    sat = request.form.get('sat')
    sat_times_taken = request.form.get('sat_times_taken')
    gpa = request.form.get('gpa')
    male = request.form.get('male')
    asian = request.form.get('asian')
    black = request.form.get('black')
    hispanic = request.form.get('hispanic')
    white = request.form.get('white')
    pacific = request.form.get('pacific')
    nativeam = request.form.get('nativeam')
    ecc = request.form.get('ecc')

    with open('../data/app/model.pkl') as f:
        my_model = pickle.load(f)
    df = pd.read_csv('../data/master.csv')
    master_cols = df.columns.values

    # Gender
    if male == '1':
        gender = 'Male'
    else:
        gender = 'Female'

    # Ethnicity
    ethnicity_vals = [asian, black, hispanic, white, pacific, nativeam]
    ethnicity_words = ['asian', 'black / african american', 'hispanic', 'white  non-hispanic', 'native hawaiian / pacific islander', 'native american']
    ethnicity = ''
    for val,word in zip(ethnicity_vals, ethnicity_words):
        if val=='1':
            ethnicity += word + '|'

    X = pd.DataFrame([[np.nan for i in xrange(len(master_cols))]], columns=master_cols)
    X['Gender'] = gender
    X['Ethnicity'] = ethnicity
    X['Highest Composite SAT Score'] = float(sat)
    X['How many times did you take the official SAT?'] = sat_times_taken
    X['High School GPA'] = float(gpa)
    X['High School Extracurricular Activities'] = ecc

    prediction = (my_model.predict_proba(X)[0][1]*100).round(1)

    return render_template('model_results.html', prediction = prediction, sat_times_taken = sat_times_taken)

@app.route('/analyzer', methods = ['GET', 'POST'])
def analyzer():
    essay = request.form.get('essay')
    similarity_tfidf = request.form.get('similarity_tfidf')
    similarity_nmf = request.form.get('similarity_nmf')
    return render_template('analyzer.html', essay = essay, similarity_tfidf = similarity_tfidf, similarity_nmf = similarity_nmf)

@app.route('/analyzer_results', methods = ['GET', 'POST'])
def analyzer_results():
    essay = request.form.get('essay')
    similarity_nmf = request.form.get('similarity_nmf')
    similarity_tfidf = request.form.get('similarity_tfidf')

    # linebreak_idx = [m.start() for m in re.finditer('\n', essay)]
    s_tokenizer = PunktSentenceTokenizer()
    sentences = s_tokenizer.tokenize(essay)
    top_sentences = textrank.summarize(essay).split('\n')
    top_idx = []
    for i,sentence in enumerate(sentences):
        if sentence in top_sentences:
            top_idx.append(i)
    sentences = list(enumerate(sentences))

    topics,similar_essays = process_essay(essay, similarity_nmf, similarity_tfidf, json_output=False)
    essay1 = similar_essays[0][1]
    essay2 = similar_essays[1][1]
    essay3 = similar_essays[2][1]
    topic1, topic2, topic3, topic4, topic5, topic6, topic7 = topics
    topic_names = ['Family', 'Music', 'Culture', 'Sport', 'Personal/Story', 'Science', 'Career']
    topic_tuples = zip(topic_names, topics)

    return render_template('analyzer_results.html', essay1 = essay1, essay2 = essay2, essay3 = essay3, topic_tuples = topic_tuples, sentences=sentences, top_idx=top_idx)

@app.route('/analyzer/api/v1.0/similar_essays', methods = ['GET', 'POST'])
def json_results():
    essay = request.form.get('essay')
    similarity_nmf = request.form.get('similarity_nmf')
    similarity_tfidf = request.form.get('similarity_tfidf')
    similar_essays = process_essay(essay, similarity_nmf, similarity_tfidf, json_output=True)

    json_output = [{'id':userid, 'essay':content} for userid,content in similar_essays]

    return jsonify({'similar_essays':json_output})


####################
## Helper functions

def process_essay(essay, similarity_nmf, similarity_tfidf, json_output):
    '''
    This method actually analyzes the essay and outputs a tuple of 3 values:
    1) list of indices for the most 'representative' sentences
    2) list of most similar essays (stored as tuples of (id,text))
    3) list of topics

    Can be used by analyzer_results() or json_results()
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
    topics = (topic1, topic2, topic3, topic4, topic5, topic6, topic7)

    # Load in database of essays and topics
    df_essay = pd.read_csv('../data/app/essays_and_topics.csv')
    essays = df_essay['content'].values
    essays_idx = df_essay['id'].values
    topic_mat = df_essay.ix[:,3:].values

    # Get essays based on NMF euclidean distance or TFIDF cosine similarity
    tm = ea.TopicModeling()
    if similarity_nmf == '1':
        similar_essays = tm.similarEssaysNMF(essay, essays, topic_mat, vectorizer, nmf)
        essay1, essay2, essay3 = similar_essays
    elif similarity_tfidf == '1':
        similar_essays = tm.similarEssaysTfidf(essay, essays, tfidf_mat, vectorizer)
        essay1, essay2, essay3 = similar_essays

    # Get most 'representative' sentences in their own essay
    # UPDATE: this has been abandoned for the superior TextRank method
    # top_idx = tm.getBestSentence(essay, vectorizer, nmf)

    if json_output:
        return similar_essays
    else:
        return (topics,similar_essays)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
