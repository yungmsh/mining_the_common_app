import model_script as ms
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
import cPickle as pickle

df = pd.read_csv('../data/train.csv', low_memory=False)
y = df.pop('top_school_final')
df_train, df_valid, y_train, y_valid = train_test_split(df, y, train_size=0.7, random_state=123)

pipeline = Pipeline([
    ('SAT', CleanSAT()),
    ('GPA', CleanGPA()),
    ('gender', Gender()),
    ('ethnicity', Ethnicity()),
    ('extracc', ExtraCurriculars()),
    ('homecountry', HomeCountry()),
    ('sports', Sports()),
    ('dummify', DummifyCategoricals()),
    ('essay_p1', CleanEssays()),
    ('essay_p2', AnalyzeEssays()),
    ('final', FinalColumns()),
    ('scale', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=10, min_samples_leaf=4,min_samples_split=2))
])

model = pipeline.fit(df_train, y_train)
y_pred = pipeline.predict(df_valid)
ms.showModelResults(y_pred, y_valid)

# with open('../app/data/model.pkl', 'w') as f:
#     pickle.dump(model, f)
