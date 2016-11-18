import model_script as ms
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
from datetime import datetime

df = pd.read_csv('../data/train.csv', low_memory=False)
y = df.pop('top_school_final')

pipeline_cat = Pipeline([
    ('gender', ms.Gender()),
    ('ethnicity', ms.Ethnicity()),
    ('extracc', ms.ExtraCurriculars()),
    ('homecountry', ms.HomeCountry()),
    ('sports', ms.Sports()),
    ('dummify', ms.DummifyCategoricals())
    # ('finalcat', ms.FinalColumnsCat())
])

pipeline_noncat = Pipeline([
    # ('essay_p1', ms.CleanEssays()),
    # ('essay_p2', ms.AnalyzeEssays()),
    ('SAT', ms.CleanSAT()),
    ('GPA', ms.CleanGPA())
    # ('finalnoncat', ms.FinalColumnsNonCat())
])

pipeline = Pipeline([
    ('noncat', pipeline_noncat),
    ('cat', pipeline_cat),
    ('finalcols', ms.FinalColumns()),
    ('scale', StandardScaler()),
    ('model', LogisticRegression())
])

params = {
    'model__C': np.logspace(-2,4,6)
    # 'model__min_samples_split': range(2,5),
    # 'model__min_weight_fraction_leaf': [0,0.03,0.05],
    # 'model__min_samples_leaf': range(1,4)
}

gs = GridSearchCV(pipeline, param_grid = params, cv=KFold(len(df),n_folds=5, shuffle=True))
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

# with open('../app/data/model.pkl', 'w') as f:
#     pickle.dump(model, f)
