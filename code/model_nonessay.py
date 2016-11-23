import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from collections import Counter, OrderedDict
import re
import time
import string
import en
from datetime import datetime
import cPickle as pickle
import exploratory_analysis as eda

start = datetime.now()

class CustomMixin(TransformerMixin):
    def get_params(self, **kwargs):
        return dict()

    def set_params(self, **kwargs):
        for key in self.get_params():
            setattr(self, key, kwargs[key])
        return self

class CleanSAT(CustomMixin):
    def fit(self, X, y):
        self.median_score = X[(X['Highest Composite SAT Score']<=2400) & (X['Highest Composite SAT Score']>=600)]['Highest Composite SAT Score'].median()
        self.median_times = X['How many times did you take the official SAT?'].median()
        return self

    def transform(self, X):
        # Remove the impossible scores first
        X['Highest Composite SAT Score'] = X['Highest Composite SAT Score'].apply(lambda x: None if x>2400 or x<600 else x)
        # Parse out the scores from the breakdown columns
        X['SAT_total_temp'] = X['Highest SAT Scores'].apply(lambda x: self.parseSAT(x) if type(x)==str else x)
        self.finalizeScores(X)
        # Give times taken a shorter name
        X['SAT_times_taken'] = X['How many times did you take the official SAT?'].copy()
        self.impute(X)

        lap = datetime.now()
        print 'Finished SAT after {} seconds.'.format((lap-start).seconds)
        return X

    def impute(self, X):
        '''
        Impute missing values
        '''
        X['SAT_total_final'].fillna(value=self.median_score, inplace=True)
        X['SAT_times_taken'].fillna(value=self.median_times, inplace=True)

    def parseSAT(self, x):
        '''
        Show logic tree in the docstring
        '''
        scores = [float(i) for i in x.split()]

        if scores[-1] > 800:
            if scores[-1] > 1600: # If the rightmost score is >1600, then it is the 2400-scale score
                return scores[-1]
            else: # The following chunk is for scores between 800 and 1600
                if sum(scores[:-1]) == scores[-1]: # If sum of breakdown == rightmost score
                    if len(scores[:-1]) == 3: # If 3 scores, must be 2400-scale
                        return scores[-1]
                    elif len(scores[:-1]) == 2: # If 2 scores, must be 1600-scale
                        return scores[-1]/1600.*2400
                    else:
                        return np.nan # Cannot be parsed
                elif sum(scores[:-1]) < scores[-1]:
                    if len(scores[:-1]) == 2: # If 2 scores, must be 2400-scale
                        return scores[-1]
                    else:
                        return np.nan # Cannot be parsed
        else: # If the rightmost score is <=800
            if len(scores) == 3: # If 3 scores, it must be 2400-scale
                return sum(scores)
            elif len(scores) == 2: # If 2 scores, it must be 1600-scale
                return sum(scores)/1600.*2400
            else:
                return np.nan # Cannot be parsed

    def finalizeScores(self, X):
        # Change NaNs to None in SAT_total_temp so we can apply a max function
        X['SAT_total_temp'] = X['SAT_total_temp'].apply(lambda x: None if str(x)=='nan' else x)
        # Take max of 2 columns
        X['SAT_total_final'] = np.max(X[['Highest Composite SAT Score','SAT_total_temp']], axis=1)
        # Remove faulty entries that don't end in '0' (SAT scores are multiples of 10)
        X['SAT_total_final'] = X['SAT_total_final'].apply(lambda x: None if x%10>0 else x)

class CleanGPA(CustomMixin):
    def fit(self, X, y):
        self.median = X[(X['High School GPA']<=4) & (X['High School GPA']>2) ]['High School GPA'].median()
        return self

    def transform(self, X):
        X['High School GPA'] = X['High School GPA'].apply(lambda x: np.nan if x>100 or x<=2 else x)
        X['High School GPA'] = X['High School GPA'].apply(lambda x: self.median if x>4 else x)
        self.impute(X)

        lap = datetime.now()
        print 'Finished GPA after {} seconds.'.format((lap-start).seconds)
        return X

    def impute(self, X):
        X['High School GPA'].fillna(value=self.median, inplace=True)

class Gender(CustomMixin):
    def fit(self, X, y):
        X['Male'] = X['Gender'].apply(lambda x: 1 if x=='Male' else 0)
        self.mode = X['Male'].mode()
        return self

    def transform(self, X):
        X['Male'] = X['Gender'].apply(lambda x: 1 if x=='Male' else 0)
        self.impute(X)

        lap = datetime.now()
        print 'Finished Gender after {} seconds.'.format((lap-start).seconds)
        return X

    def impute(self, X):
        X['Male'].fillna(value=self.mode, inplace=True)

class Ethnicity(CustomMixin):
    def fit(self, X, y):
        self.ethnicity_cols = ['Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Hispanic', 'Ethnicity_White', 'Ethnicity_Pacific', 'Ethnicity_NativeAm']
        self.ethnicity_words = ['asian', 'black / african american', 'hispanic', 'white non-hispanic', 'native hawaiian / pacific islander', 'native american']
        return self

    def transform(self, X):
        self.extract(X)
        self.impute(X)

        lap = datetime.now()
        print 'Finished Ethnicity after {} seconds.'.format((lap-start).seconds)
        return X

    def impute(self, X):
        for col in self.ethnicity_cols:
            X[col].fillna(value=0, inplace=True)

    def extract(self, X):
        X['Ethnicity2'] = X['Ethnicity'].apply(lambda x: self.cleanEthnicity(x) if type(x)==str else x)
        for col,word in zip(self.ethnicity_cols, self.ethnicity_words):
            X[col] = X['Ethnicity2'].apply(lambda x: 1 if type(x)==list and word in x else 0)

    def cleanEthnicity(self, x):
        x = x.lower().split('|')
        if 'prefer not to share' in x:
            del x[x.index('prefer not to share')]
        if len(x) == 0:
            return np.nan
        else:
            return x

class ExtraCurriculars(CustomMixin):
    def fit(self, X, y):
        leader_words = ['leader','president','founder']
        arts_words = ['arts', 'music', 'jazz', 'band', 'orchestra', 'choir', 'drama', 'theater']
        award_words = ['award', 'scholarship', 'achievement', 'prize']
        community_words = ['volunteer', 'community','cleanup', 'ngo', 'environment', 'humanity','green', 'charity']
        academic_words = ['science', 'math', 'engineering']
        gov_words = ['debate', 'model', 'government']
        diversity_words = ['alliance', 'multicultural', 'diversity']
        race_words = ['naacp','asian','jewish','german','french','japanese','italian','chinese']
        self.lst_of_keywords = [leader_words, arts_words, award_words, community_words, academic_words, gov_words, diversity_words, race_words]
        self.cols = ['leader', 'arts', 'award', 'community', 'academic', 'gov', 'diversity', 'race_ecc']
        return self

    def transform(self, X):
        for col,lst in zip(self.cols, self.lst_of_keywords):
            X[col] = X['High School Extracurricular Activities'].apply(lambda x: self.extract(x, lst))
        lap = datetime.now()
        print 'Finished ECC after {} seconds.'.format((lap-start).seconds)
        return X

    def impute(self, X):
        for col in self.cols:
            X[col].fillna(value=0, inplace=True)

    def extract(self, x, lst):
        '''
        INPUT:
        OUTPUT:
        '''
        if type(x)==str:
            x = x.lower()
            for word in lst:
                if x.find(word)>-1:
                    return 1
            return 0
        else:
            return 0

class HomeCountry(CustomMixin):
    def fit(self, X, y):
        self.extract(X)
        self.mode = X['Home Country_US'].mode()
        return self

    def transform(self, X):
        self.extract(X)
        self.impute(X)

        lap = datetime.now()
        print 'Finished HomeCountry after {} seconds.'.format((lap-start).seconds)
        return X

    def impute(self, X):
        X['Home Country_US'].fillna(self.mode)

    def extract(self, X):
        regex = re.compile('[[]\S+[]]\s[[]\S+[]]\s(.+)')
        get_country = lambda x: regex.findall(x)[0] if not x is np.nan and not x is None and len(regex.findall(x))>0 else x
        X['Home Country'] = X['Home Country'].apply(get_country)
        X['Home Country_US'] = X['Home Country'].apply(lambda x: 1 if x=='United States' else 0)

class Sports(CustomMixin):
    def fit(self, X, y):
        all_sports = list(X['High School Sports Played'].apply(eda.getAllSports))
        self.unique_sports = eda.getUniqueSports(all_sports)
        return self

    def transform(self, X):
        # Initialize dummy variables for each sport category, set to 0s.
        # for sport in self.unique_sports:
        #     X['sports_'+sport] = 0
        # Fill in the dummy variables for each sport category (1 or 0).
        # eda.parseSports(X, self.unique_sports)

        regexp = re.compile('[[]\S+[]]\s[[]\S+[]]\s(.+)')
        # Create a sportsVarsity dummy variable
        X['sportsVarsity'] = X['High School Sports Played'].apply(lambda x: eda.parseVarsity(x, self.unique_sports, regexp))
        # Create a sportsCaptain dummy variable.
        X['sportsCaptain'] = X['High School Sports Played'].apply(lambda x: eda.parseCaptain(x, self.unique_sports, regexp))

        lap = datetime.now()
        print 'Finished Sports after {} seconds.'.format((lap-start).seconds)
        return X

class DummifyCategoricals(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = pd.get_dummies(X, columns=['Academic Performance in High School'], prefix='HS')

        lap = datetime.now()
        print 'Finished Dummify after {} seconds.'.format((lap-start).seconds)
        return X

class FinalColumns(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        final_cols = ['SAT_total_final', 'SAT_times_taken', 'High School GPA', 'Male', 'leader', 'arts', 'award', 'community', 'academic', 'gov', 'diversity', 'race_ecc', 'Home Country_US']

        ethnicity_cols = [col for col in X.columns if col.find('Ethnicity_')>-1]
        HS_perf_cols = [col for col in X.columns if col.find('HS_')>-1]
        sports_cols = ['sportsVarsity', 'sportsCaptain']

        final_cols.extend(ethnicity_cols)
        final_cols.extend(HS_perf_cols)
        final_cols.extend(sports_cols)

        good_cols = ['SAT_total_final', 'SAT_times_taken', 'High School GPA', 'Male', 'leader', 'award', 'academic', 'gov', 'sportsVarsity', 'sportsCaptain', 'Ethnicity_Black', 'Ethnicity_White', 'HS_Steady']

        X_model = X[final_cols].copy()
        print X_model.isnull().sum()

        lap = datetime.now()
        print 'Finished FinalColumns after {} seconds.'.format((lap-start).seconds)
        return X_model
