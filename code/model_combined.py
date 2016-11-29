import model_nonessay as mne
import model_essay as me
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
from datetime import datetime

def refreshData(train=True):
    if train:
        df = pd.read_csv('../data/train.csv', low_memory=False)
    else:
        df = pd.read_csv('../data/test.csv', low_memory=False)
    y = df.pop('top_school_final')
    return df, y

def mainPipeline(model, main_or_essay='main'):
    if main_or_essay == 'main':
        final_step = ('finalcols', mne.FinalColumns())
    elif main_or_essay == 'essay':
        final_step = ('finalcols', mne.FinalColumnsWithEssay())

    p = Pipeline([
        ('SAT', mne.CleanSAT()),
        ('GPA', mne.CleanGPA()),
        ('gender', mne.Gender()),
        ('ethnicity', mne.Ethnicity()),
        ('extracc', mne.ExtraCurriculars()),
        ('homecountry', mne.HomeCountry()),
        ('sports', mne.Sports()),
        ('dummify', mne.DummifyCategoricals()),
        final_step,
        # ('scale', StandardScaler()),
        ('model', model)
    ])
    return p


class DualPipeline(object):
    def __init__(self):
        pass

    def fit(self, X, y, p1, p2, params1, params2, scoring):
        '''
        INPUT: X (Pandas dataframe), y (np.array), p1 (sklearn Pipeline), p2 (sklearn Pipeline), params1 (dict), params2 (dict)
        OUTPUT: None

        Performs grid search on two separate pipelines.
        '''
        # Pipeline 1
        # df,y = refreshData()
        self.gs1 = GridSearchCV(p1, param_grid = params1, cv=5, scoring=scoring)
        self.gs1.fit(X, y)

        # Pipeline 2
        # df,y = refreshData()
        self.gs2 = GridSearchCV(p2, param_grid = params2, cv=5, scoring=scoring)
        self.gs2.fit(X, y)

    def predict(self, X, method='avg', model=None, proba=True):
        '''
        INPUT: X (Panda dataframe), method (string), model (sklearn model object), proba (bool)
        OUTPUT: y_proba (list)

        Predicts two lists of probabilities (outputs of two classifiers), then outputs a list of new probabilities.

        Final prediction method can be specified:
        'avg': calculates the mean of the two probabilities
        'model': runs the probabilities as new features of another model to predict y.

        Prediction can be probabilities or classes, specified by the parameter 'proba' (True/False).
        '''
        y1_proba = self.gs1.predict_proba(X)[:,1]
        y2_proba = self.gs2.predict_proba(X)[:,1]
        if method=='avg':
            y_proba = (y1_proba + y2_proba) / 2.
            y_pred = np.round(y_proba)
        elif method=='model':
            y1_proba = y1_proba.reshape(-1,1)
            y2_proba = y2_proba.reshape(-1,1)
            X = np.hstack((y1_proba, y2_proba))
            model.fit(X, y)
            y_proba = model.predict_proba(X)[:,1]
            y_pred = model.predict(X)

        if proba:
            return y_proba
        else:
            return y_pred

class EssayPipeline(object):
    def __init__(self):
        self.clean = me.CleanEssays()
        self.analyze = me.AnalyzeEssays()

    def findEssayIdx(self, X):
        '''
        Helper function to identify essay indices as an initial step in both fitting and predicting.
        The idea is we're modeling a subset of the full dataset so we only want to choose the appropriate indices.
        '''
        self.clean = me.CleanEssays()
        self.clean.cleanEverything(X)
        self.essay_idx = self.clean.essay_idx

    def fit(self, X, y, model, params, scoring):
        self.findEssayIdx(X)
        X_mod = X.loc[self.essay_idx,:].copy()
        y = y[self.essay_idx].copy()
        self.analyze.fit(X_mod)
        X_mod = self.analyze.transform(X_mod)
        # self.essay_cols = ['5000_words_frac', 'essay_topic1', 'essay_topic2', 'essay_topic3', 'essay_topic4', 'essay_topic5', 'essay_topic6', 'essay_topic7']
        pipeline = mainPipeline(model, main_or_essay='essay')
        self.gs = GridSearchCV(pipeline, params, cv=3, scoring=scoring)
        self.gs.fit(X_mod, y)
        # self.gs.fit(X_mod.loc[:,self.essay_cols], y)

    def predict(self, X):
        self.findEssayIdx(X)
        X_mod = X.loc[self.essay_idx,:].copy()
        X_mod = self.analyze.transform(X_mod)
        y_proba = self.gs.predict_proba(X_mod)[:,1]
        # y_proba = self.gs.predict_proba(X_mod.loc[:,self.essay_cols])[:,1]
        return y_proba

class GrandModel(object):
    def __init__(self):
        pass

    def fit(self, X, y, model1, model2, params1, params2, essay_model=None, essay_params=None, scoring='precision'):
        '''
        Do a fit on both DualPipeline and EssayPipeline.
        '''
        # DualPipeline
        p1 = mainPipeline(model1)
        p2 = mainPipeline(model2)
        self.dp = DualPipeline()
        self.dp.fit(X, y, p1, p2, params1, params2, scoring)

        # EssayPipeline
        self.ep = EssayPipeline()
        # df,y = refreshData()
        self.ep.fit(X, y, essay_model, essay_params, scoring)

    def predict(self, X, proba=False):
        y_proba_main = self.dp.predict(X)
        y_proba_essay = self.ep.predict(X)
        if not proba:
            y_pred = self._combinePredictions(y_proba_main, y_proba_essay, self.ep.essay_idx, proba=False)
            return y_pred
        else:
            y_proba = self._combinePredictions(y_proba_main, y_proba_essay, self.ep.essay_idx, proba=True)
            return y_proba

    def _combinePredictions(self, y_proba_main, y_proba_essay, essay_idx, proba=False):
        '''
        INPUT: y_proba_main (list of probabilities), y_proba_essay (list of probabilities), essay_idx (list of indices)
        OUTPUT: y_final (list of predictions)

        Combines the predictions from the non-essay model with the essay model.

        Notes:
        len(y_essay) = len(essay_idx)
        len(y_main) = len(y_final)
        '''
        y_proba_final = y_proba_main.copy()
        for i,j in enumerate(np.array(essay_idx)):
            y_proba_final[j] = np.mean((y_proba_main[j], y_proba_essay[i]))
        y_final = np.round(y_proba_final)

        if not proba:
            return y_final
        else:
            return y_proba_final

    def showModelResults(self, y, y_pred):
        metric_text = ['Accuracy','Precision','Recall','Confusion Matrix']
        metrics = [accuracy_score, precision_score, recall_score, confusion_matrix]
        for text,score in zip(metric_text, metrics):
            print '{}: {}'.format(text, score(y, y_pred))

def showLogisticCoefs(features, coefs):
    for coef,feature in sorted(zip(np.exp(coefs), features), reverse=True):
        print feature, np.round(coef,4)

def getROC(y_true, y_proba, show_area=False):
    '''
    Get a tuple of FPR and TPR to plot in an ROC curve.
    '''
    roc = roc_curve(y_true, y_proba)
    fpr, tpr = roc[0], roc[1]
    if show_area:
        area = auc(fpr, tpr)
        return fpr, tpr, area
    else:
        return fpr, tpr

def plotROC(tuples, labels, show_area=False):
    '''
    INPUT: tuples (list of tuples), labels (list of strings)
    OUTPUT: plot
    '''
    if not show_area:
        for tup, label in zip(tuples,labels):
            fpr, tpr = tup[0], tup[1]
            plt.plot(fpr, tpr, lw=2, label=label)
    else:
        for tup, label in zip(tuples,labels):
            fpr, tpr, area = tup[0], tup[1], tup[2]
            plt.plot(fpr, tpr, lw=2, label='{} [AUC = {}]'.format(label, np.round(area,3)))
    plt.legend(loc='lower right')

if __name__ == '__main__':
    df,y = refreshData()

    m1 = LogisticRegression(n_jobs=-1)
    m2 = RandomForestClassifier(n_jobs=-1)
    essay_model = RandomForestClassifier(n_jobs=-1)

    params1 = {
        'model__C': [1000]
    }
    params2 = {
        'model__min_samples_split': [4]
    }
    essay_params = {
        'model__min_samples_split': range(2,3)
        # 'model__min_weight_fraction_leaf': [0,0.01],
        # 'model__min_samples_leaf': range(1,3)
    }

    gm = mc.GrandModel()
    gm.fit(df, y, m1, m2, params1, params2, essay_model, essay_params, scoring='precision')
    y_pred = gm.predict(df)
    print gm.showModelResults(y, y_pred)

    with open('../app/data/model.pkl', 'w') as f:
        pickle.dump(gm, f)

    # THE VERY LAST THING TO DO (new script?)
    # df_test = pd.read_csv('../data/test.csv', low_memory=False)
    # y_test = df_test.pop('top_school_final')
    # y_pred = gm.predict(df_test)
    # print gm.showModelResults(y_test, y_pred)
