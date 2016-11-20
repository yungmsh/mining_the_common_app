import model_script as ms
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

pipeline1 = Pipeline([
    # ('essay_p1', ms.CleanEssays()),
    # ('essay_p2', ms.AnalyzeEssays()),
    ('SAT', ms.CleanSAT()),
    ('GPA', ms.CleanGPA()),
    ('gender', ms.Gender()),
    ('ethnicity', ms.Ethnicity()),
    ('extracc', ms.ExtraCurriculars()),
    ('homecountry', ms.HomeCountry()),
    ('sports', ms.Sports()),
    ('dummify', ms.DummifyCategoricals()),
    ('finalcols', ms.FinalColumns()),
    ('scale', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline2 = Pipeline([
    ('essay_p1', ms.CleanEssays()),
    ('essay_p2', ms.AnalyzeEssays()),
    ('finalcols', ms.FinalEssayColumns()),
    ('model', KNeighborsClassifier())
])

def ensembleClassifier(p1, p2, method='avg', model = None):
    '''
    Takes two Pipeline objects, turns them into two lists of probabilities (outputs of two classifiers), then outputs a final prediction.

    Final prediction method can be specified:
    'avg': calculates the mean of the two probabilities
    'model': runs the probabilities as new features of another model to predict y.
    '''
    df, y = refreshData()
    p1.fit(df, y)
    y1_prob = p1.predict_proba(df)[:,1]
    df, y = refreshData()
    p2.fit(df, y)
    y2_prob = p2.predict_proba(df)[:,1]

    if method=='avg':
        y_pred = np.round((y1_prob + y2_prob) / 2.)
    elif method=='model':
        y1_prob = y1_prob.reshape(-1,1)
        y2_prob = y2_prob.reshape(-1,1)
        X = np.hstack((y1_prob, y2_prob))
        print y1_prob.shape, y2_prob.shape
        print X.shape
        model.fit(X, y)
        y_pred = model.predict(X)

    return y_pred

params_rf = {
    'model__min_samples_split': range(2,5),
    'model__min_weight_fraction_leaf': [0,0.03,0.05],
    'model__min_samples_leaf': range(1,4)
}

params_lr = {
    'model__C': np.logspace(-2,4,6)
}

gs = GridSearchCV(pipeline1, param_grid = params_lr, cv=KFold(len(df),n_folds=3, shuffle=True))
gs.fit(df, y)
y_pred = gs.predict(df)
ms.showModelResults(y_pred, y)

# kf = KFold(len(df), n_folds=5, shuffle=True)
# for train_index, test_index in kf:
#     X_train, X_test = df.iloc[train_index,:], df.iloc[test_index,:]
#     y_train, y_test = y[train_index], y[test_index]
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     ms.showModelResults(y_pred, y_test)

# y_pred = pipeline.predict(df)
# ms.showModelResults(y_pred, y)

# if __name__ == '__main__':
# df,y = refreshData()
# with open('../app/data/model.pkl', 'w') as f:
#     pickle.dump(model, f)
