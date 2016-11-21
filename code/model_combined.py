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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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

class DualPipeline(object):
    def __init__(self):
        pass

    def fit(self, X, y, p1, p2, params1, params2):
        '''
        INPUT: X (Pandas dataframe), y (np.array), p1 (sklearn Pipeline), p2 (sklearn Pipeline), params1 (dict), params2 (dict)
        OUTPUT: None

        Performs grid search on two separate pipelines.
        '''
        # Pipeline 1
        # df,y = refreshData()
        self.gs1 = GridSearchCV(p1, param_grid = params1, cv=KFold(len(X),n_folds=2, shuffle=True))
        self.gs1.fit(X, y)

        # Pipeline 2
        # df,y = refreshData()
        self.gs2 = GridSearchCV(p2, param_grid = params2, cv=KFold(len(X),n_folds=2, shuffle=True))
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
        The idea is we're modeling a subset of the full dataset, so we only want to choose the appropriate indices.
        '''
        self.clean = me.CleanEssays()
        self.clean.cleanEverything(X)
        self.essay_idx = self.clean.essay_idx

    def fit(self, X, y, model):
        self.findEssayIdx(X)
        X = X.loc[self.essay_idx,:].copy()
        y = y[self.essay_idx].copy()
        self.analyze.fit(X)
        X = self.analyze.transform(X)
        self.essay_cols = ['5000_words_frac', 'essay_topic1', 'essay_topic2', 'essay_topic3', 'essay_topic4', 'essay_topic5', 'essay_topic6', 'essay_topic7']
        self.model = model
        self.model.fit(X.loc[:,self.essay_cols], y)

    def predict(self, X):
        self.findEssayIdx(X)
        X = X.loc[self.essay_idx,:]
        X = self.analyze.transform(X)
        print X.columns
        y_proba = self.model.predict_proba(X.loc[:,self.essay_cols])[:,1]
        return y_proba

def combinePredictions(y_proba_main, y_proba_essay, essay_idx):
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
    return y_final

class GrandModel(object):
    def __init__(self):
        pass

    def fit(self, X, y, model1, model2, params1, params2, essay_model):
        # DualPipeline
        p1 = mainPipeline(model1)
        p2 = mainPipeline(model2)
        self.dp = DualPipeline()
        self.dp.fit(X, y, p1, p2, params1, params2)

        # EssayPipeline
        self.ep = EssayPipeline()
        # df,y = refreshData()
        self.ep.fit(X, y, essay_model)

    def predict(self, X):
        y_proba_main = self.dp.predict(X)
        y_proba_essay = self.ep.predict(X)
        y_pred = self._combinePredictions(y_proba_main, y_proba_essay, self.ep.essay_idx)
        return y_pred

    def _combinePredictions(self, y_proba_main, y_proba_essay, essay_idx):
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
        return y_final

    def showModelResults(self, y, y_pred):
        metric_text = ['Accuracy','Precision','Recall','Confusion Matrix']
        metrics = [accuracy_score, precision_score, recall_score, confusion_matrix]
        for text,score in zip(metric_text, metrics):
            print '{}: {}'.format(text, score(y, y_pred))
        # print 'Accuracy:', accuracy_score(y_pred, y_true)
        # print 'Precision:', precision_score(y_pred, y_true)
        # print 'Recall:', recall_score(y_pred, y_true)
        # print 'Confusion Matrix: ', confusion_matrix(y_pred, y_true)

# ORIGINAL FUNCTION BELOW
# def runModel(model1, model2, params1, params2, essay_model):
#     # DualPipeline fit and predict
#     p1 = mainPipeline(model1)
#     p2 = mainPipeline(model2)
#     dp = DualPipeline()
#     dp.fit(p1, p2, params1, params2)
#     y_proba_main = dp.predict(df)
#
#     # EssayPipeline fit and predict
#     ep = EssayPipeline()
#     df,y = refreshData()
#     ep.fit(df, y, essay_model)
#     y_proba_essay = ep.predict(df)
#
#     # Combine predictions
#     y_pred = combinePredictions(y_proba_main, y_proba_essay, essay_idx)
#     return y_pred

if __name__ == '__main__':
    df,y = refreshData()

    p1 = LogisticRegression()
    p2 = RandomForestClassifier()
    params1 = {
        'model__C': np.logspace(-2,4,2)
    }
    params2 = {
        'model__min_samples_split': range(2,3),
        # 'model__min_weight_fraction_leaf': [0,0.03,0.05],
        # 'model__min_samples_leaf': range(1,4)
    }
    essay_model = KNeighborsClassifier()

    gm = GrandModel()
    gm.fit(df, y, p1, p2, params1, params2, essay_model)
    y_pred = gm.predict(df)
    print gm.showModelResults(y, y_pred)

    # with open('../app/data/model.pkl', 'w') as f:
    #     pickle.dump(gm, f)

    # THE VERY LAST THING TO DO (new script?)
    # df_test = pd.read_csv('../data/test.csv', low_memory=False)
    # y_test = df_test.pop('top_school_final')
