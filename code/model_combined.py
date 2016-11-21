import model_main as mm
import model_essay as me
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
from datetime import datetime

def refreshData():
    df = pd.read_csv('../data/train.csv', low_memory=False)
    y = df.pop('top_school_final')
    return df, y

def mainPipeline(model):
    p = Pipeline([
        ('SAT', mm.CleanSAT()),
        ('GPA', mm.CleanGPA()),
        ('gender', mm.Gender()),
        ('ethnicity', mm.Ethnicity()),
        ('extracc', mm.ExtraCurriculars()),
        ('homecountry', mm.HomeCountry()),
        ('sports', mm.Sports()),
        ('dummify', mm.DummifyCategoricals()),
        ('finalcols', mm.FinalColumns()),
        ('scale', StandardScaler()),
        ('model', model)
    ])
    return p

def essayPipeline(model):
    '''
    INPUT: model (sklearn model object)
    OUTPUT: y (list), y_pred (list), essay_idx (list)

    Custom pipeline for building a separate model using essay features as inputs to the model.
    '''
    df,y = refreshData()
    ce = me.CleanEssays()
    ae = me.AnalyzeEssays()
    essay_idx, df = ce.getEssayIndices(df)
    df = df.loc[essay_idx,:].copy()
    y = y[essay_idx].copy()

    ae.fit(df)
    df = ae.transform(df)

    essay_cols = ['5000_words_frac', 'essay_topic1', 'essay_topic2', 'essay_topic3', 'essay_topic4', 'essay_topic5', 'essay_topic6', 'essay_topic7']
    model.fit(df.loc[:,essay_cols], y)
    y_proba = model.predict_proba(df.loc[:,essay_cols])

    return y_proba[:,1], essay_idx

def ensembleModel(p1, p2, params1, params2, method='avg', model = None, proba = True):
    '''
    INPUT: p1 (sklearn Pipeline), p2 (``), method (string), model (sklearn model object)
    OUTPUT: y_proba (list)

    Takes two Pipeline objects, turns them into two lists of probabilities (outputs of two classifiers), then outputs a list of new probabilities.

    Final prediction method can be specified:
    'avg': calculates the mean of the two probabilities
    'model': runs the probabilities as new features of another model to predict y.
    '''
    df, y = refreshData()
    y1_proba = runGridSearch(df, y, p1, params1)
    # p1.fit(df, y)
    # y1_proba = p1.predict_proba(df)[:,1]
    df, y = refreshData()
    y2_proba = runGridSearch(df, y, p2, params2)
    # p2.fit(df, y)
    # y2_proba = p2.predict_proba(df)[:,1]

    if method=='avg':
        y_proba = (y1_proba + y2_proba) / 2.
        y_pred = np.round(y_proba)
    elif method=='model':
        y1_proba = y1_proba.reshape(-1,1)
        y2_proba = y2_proba.reshape(-1,1)
        X = np.hstack((y1_proba, y2_proba))
        # print y1_proba.shape, y2_proba.shape
        # print X.shape
        model.fit(X, y)
        y_proba = model.predict_proba(X)[:,1]
        y_pred = model.predict(X)

    if proba:
        return y_proba
    else:
        return y_pred

def runGridSearch(df, y, p, params, proba=True):
    gs = GridSearchCV(p, param_grid = params, cv=KFold(len(df),n_folds=5, shuffle=True))
    gs.fit(df, y)
    y_proba = gs.predict_proba(df)[:,1]
    y_pred = gs.predict(df)
    if proba:
        return y_proba
    else:
        return y_pred

def combinePredictions(y_main, y_essay, essay_idx):
    '''
    INPUT: y_main (long list), y_essay (short list), essay_idx (list)
    OUTPUT: y_final (list)

    Combines the predictions from the non-essay model with the essay model.

    Notes:
    len(y_essay) = len(essay_idx)
    len(y_main) = len(y_final)
    '''
    y_final = y_main.copy()
    for i,j in enumerate(np.array(essay_idx)):
        y_final[j] = np.mean((y_main[j], y_essay[i]))
    return y_final

params_rf = {
    'model__min_samples_split': range(2,5),
    # 'model__min_weight_fraction_leaf': [0,0.03,0.05],
    # 'model__min_samples_leaf': range(1,4)
}

params_lr = {
    'model__C': np.logspace(-2,4,2)
}

def runModel():
    y_essay_proba, essay_idx = essayPipeline(RandomForestClassifier())
    pipe_lr = mainPipeline(LogisticRegression())
    pipe_rf = mainPipeline(RandomForestClassifier())
    y_main_proba = ensembleModel(pipe_lr, pipe_rf, params_lr, param_rf, method='avg')

    y_proba = combinePredictions(y_main_proba, y_essay_proba, essay_idx)
    y_pred = np.round(y_proba)
    return y_pred

if __name__ == '__main__':
    y_pred = runModel()
    ms.showModelResults(y_pred, y)

    # with open('../app/data/model.pkl', 'w') as f:
    #     pickle.dump(model, f)
