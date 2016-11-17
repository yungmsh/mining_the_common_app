import model_script as ms
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
from datetime import datetime

df = pd.read_csv('../data/train.csv', low_memory=False)
y = df.pop('top_school_final')

pipeline = Pipeline([
    ('SAT', ms.CleanSAT()),
    ('GPA', ms.CleanGPA()),
    ('gender', ms.Gender()),
    ('ethnicity', ms.Ethnicity()),
    ('extracc', ms.ExtraCurriculars()),
    ('homecountry', ms.HomeCountry()),
    ('sports', ms.Sports()),
    ('dummify', ms.DummifyCategoricals()),
    ('essay_p1', ms.CleanEssays()),
    ('essay_p2', ms.AnalyzeEssays()),
    ('final', ms.FinalColumns()),
    ('scale', StandardScaler()),
    ('model', RandomForestClassifier())
])

params = {
    'model__min_samples_split': range(2,5),
    'model__min_weight_fraction_leaf': [0,0.03,0.05],
    'model__min_samples_leaf': range(1,4)
}

start = datetime.now()
gs = GridSearchCV(pipeline, param_grid = params, cv=KFold(len(df),n_folds=5, shuffle=True))
gs.fit(df, y)
end = datetime.now()
print "Finished modeling after {} seconds".format((end-start).seconds)
print "Best estimator is:"
print gs.best_estimator_

y_pred = gs.predict(df)
ms.showModelResults(y_pred, y)
# with open('../app/data/model.pkl', 'w') as f:
#     pickle.dump(model, f)
