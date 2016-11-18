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

                if sum(scores[:-1]) == scores[-1]: # If the sum of breakdown == rightmost score
                    if len(scores[:-1]) == 3: # If breakdown has 3 scores, it must be 2400-scale
                        return scores[-1]
                    elif len(scores[:-1]) == 2: # If breakdown has 2 scores, it must be 1600-scale
                        return scores[-1]/1600.*2400
                    else:
                        return np.nan # Cannot be parsed

                elif sum(scores[:-1]) < scores[-1]: # If the sum of breakdown < rightmost score
                    if len(scores[:-1]) == 2: # If the breakdown has 2 scores, it should be 2400-scale.
                        return scores[-1]
                    else:
                        return np.nan # Cannot be parsed

        else: # If the rightmost score is <=800
            if len(scores) == 3: # If the breakdown has 3 scores, it must be 2400-scale
                return sum(scores)
            elif len(scores) == 2: # If the breakdown has 2 scores, it must be 1600-scale
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

class CleanEssays(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        self.updateEssayCols(X)
        self.updateWordCounts(X)
        regexp1 = re.compile('Full Length Personal Statement([\s\S]*)')
        regexp2 = re.compile('[[]\d+[]]')
        X['essay_c3_edit'] = X['essay_c3'].apply(lambda x: self.cleanEssayC3(x, regexp1, regexp2) if not x is np.nan else x)

        for old,new in zip(['essay_c1', 'essay_c2'], ['essay_c1_edit', 'essay_c2_edit']):
            X[new] = X.apply(lambda x: x[old] if x['Undergraduate Personal Statement Type'] == 'Full Length Personal Statement' else np.nan, axis=1)
        self.new_cols = [col+'_edit' for col in self.cols]
        self.removeASCII(X)
        self.removeExtremes(X)
        self.removeOverlaps(X)
        X['essay_final'] = X.apply(self.consolidateEssays, axis=1)

        lap = datetime.now()
        print 'Finished CleanEssays after {} seconds.'.format((lap-start).seconds)

        # final_cols = ['Male', 'leader', 'award', 'academic', 'gov', 'sportsVarsity', 'sportsCaptain', 'Ethnicity_Black', 'Ethnicity_White', 'HS_Steady', 'SAT_total_final', 'SAT_times_taken', 'High School GPA']
        # print "Nulls in each column:"
        # print X[final_cols].isnull().sum()

        return X

    def updateEssayCols(self, X):
        '''
        INPUT: X (DataFrame)
        OUTPUT: None

        Updates the three essay columns to a simpler form, then drops the old columns.
        '''
        old_cols = ['Undergraduate Personal Statement', 'Undergraduate Essay Details', 'NEW Personal Statement']
        self.cols = ['essay_c'+str(i+1) for i,v in enumerate(old_cols)]
        for old,new in zip(old_cols,self.cols):
            X[new] = X[old].copy()

    def updateWordCounts(self, X):
        '''
        INPUT: X (DataFrame)
        OUTPUT: None

        Creates/Updates columns that show the word count for the three essay columns.
        '''
        self.wordcnt_cols = ['wordcnt_'+col for col in self.cols]
        for wordcnt_col, col in zip(self.wordcnt_cols, self.cols):
            X[wordcnt_col] = X[col].apply(lambda x: len(x.split()) if not x is np.nan and not x == None else x)

    def cleanEssayC3(self, essay, regexp1, regexp2):
        '''
        INPUT: essay (string)
        OUTPUT: cleaned essay (string)

        Receives an unformatted chunk of text, and extracts just the essay part.
        '''
        content = regexp1.findall(essay)
        if len(content)>0:
            cleaned = content[0].strip()
            brackets = regexp2.findall(cleaned)
            if len(brackets)>0:
                pos = cleaned.find(brackets[0])
                return cleaned[:pos].strip()
        else:
            return np.nan

    def removeASCII(self, X):
        '''
        INPUT: X (DataFrame), cols (list)
        OUTPUT: None

        Given a dataframe and list of essay cols, this function removes ASCII characters in each entry in each col.
        '''
        regexp = re.compile('[^\w\s\d,.-]+')
        for col in self.cols:
            X[col] = X[col].apply(lambda x: self._ASCII(x, regexp) if not x is np.nan and not x is None else x)

    def _ASCII(self, essay, regexp):
        '''
        Internal function for removeASCII function above
        '''
        for exp in set(regexp.findall(essay)):
            essay = essay.replace(exp, '')
        return essay

    def removeExtremes(self, X):
        '''
        INPUT: df (DataFrame), cols (list)
        OUTPUT: None
        '''
        for col in self.new_cols:
            X[col] = X[col].apply(lambda x: x if not x is np.nan and not x is None and len(x.split())>200 and len(x.split())<1100 else np.nan)

    def removeOverlaps(self, X):
        '''
        INPUT: df (DataFrame), cols (list of cols), keep_col (string)
        OUTPUT: None
        '''
        remove_col = 'essay_c2_edit'
        idx = []
        for col in self.new_cols:
            idx.append(X[X[col].notnull()==True].index)
        intersect = np.intersect1d(idx[0],idx[1])
        X.loc[intersect, remove_col] = np.nan

    def consolidateEssays(self, X):
        '''
        INPUT: df (DataFrame), cols (list of cols to consolidate)
        OUTPUT: essay text (string) or np.nan
        '''
        c1,c2,c3 = self.new_cols
        if type(X[c1])==str:
            return X[c1]
        elif type(X[c2])==str:
            return X[c2]
        elif type(X[c3])==str:
            return X[c3]
        else:
            return np.nan

class AnalyzeEssays(CustomMixin):
    def fit(self, X, y):
        # Get SAT words loaded first
        self.getSATWords()
        # Preprocess: remove stopwords and perform stemming
        essays = self.preprocess(X, 'fit')

        # Vectorize using tfidf
        self.vec = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, max_features=10000)
        self.vec.fit(essays)
        mat = self.vec.transform(essays)
        print 'Finished TFIDF vectorizing on fit step'

        # Use NMF to perform topic modeling
        self.nmf = NMF(n_components=7, random_state=123)
        self.nmf.fit(mat)
        mat_nmf = self.nmf.transform(mat)
        print 'Finished NMF on fit step'
        self.essay_topics = ['essay_topic1', 'essay_topic2', 'essay_topic3', 'essay_topic4', 'essay_topic5', 'essay_topic6', 'essay_topic7']
        df_nmf = pd.DataFrame(mat_nmf, columns = self.essay_topics)

        # Calculate 'avg' values of topics (to impute missing essays later)
        self.avg_topics = df_nmf.mean().values

        # Calculate mean of SAT word cols
        self.words_1000_mean = X['1000_words_cnt'].mean()
        self.words_5000_mean = X['5000_words_frac'].mean()

        return self

    def transform(self, X):
        essays = self.preprocess(X, 'transform')
        mat = self.vec.transform(essays)
        mat_nmf = self.nmf.transform(mat)
        print len(X)
        print len(mat_nmf)
        df_nmf = pd.DataFrame(mat_nmf, columns = self.essay_topics)
        X = X.join(df_nmf)

        # Impute missing values with 'avg' value for each topic
        for col,value in zip(self.essay_topics, self.avg_topics):
            X[col].fillna(value=value, inplace=True)

        # Impute missing values for SAT word cols
        X['1000_words_cnt'].fillna(value=self.words_1000_mean, inplace=True)
        X['5000_words_frac'].fillna(value=self.words_5000_mean, inplace=True)

        print 'Finished imputing essay features (topics AND fraction of SAT words)'
        lap = datetime.now()
        print 'Finished Analyze Essays after {} seconds.'.format((lap-start).seconds)

        # final_cols = ['Male', 'leader', 'award', 'academic', 'gov', 'sportsVarsity', 'sportsCaptain', 'Ethnicity_Black', 'Ethnicity_White', 'HS_Steady', '5000_words_frac', 'essay_topic1', 'essay_topic2', 'essay_topic3', 'essay_topic4', 'essay_topic5', 'essay_topic6', 'essay_topic7']
        # print "Nulls in each column:"
        # print X[final_cols].isnull().sum()
        return X

    def preprocess(self, X, fit_or_transform):
        # If fitting, just use non-null values
        # if fit_or_transform == 'fit':
        #     essays = X[X['essay_final'].notnull()]['essay_final'].values
        # # If transforming, use all values
        # elif fit_or_transform == 'transform':
        essays = X['essay_final'].values

        punct = string.punctuation
        stop_words = stopwords.words('english')
        wn = WordNetLemmatizer()

        for i,essay in enumerate(essays):
            if not essay in (np.nan, None):
                try:
                    essay = essay.encode('ascii','ignore')
                except:
                    try:
                        essay = essay.decode('ascii','ignore')
                    except:
                        pass
                # Correct tense and lemmatize
                essay = essay.split()
                for i,word in enumerate(essay):
                    try:
                        essay[i] = wn.lemmatize(en.verb.present(word.split(punct)))
                    except:
                        pass
                # Extract data for SAT words
                c = Counter(essay)
                count_1000 = sum([c[word] for word in self.words_1000])
                count_5000 = sum([c[word] for word in self.words_5000])
                X.loc[i, '1000_words_cnt'] = count_1000
                if len(essay)==0:
                    X.loc[i,'5000_words_frac'] = 0
                else:
                    X.loc[i, '5000_words_frac'] = count_5000 / float(len(essay))

                # Remove stop words
                essay = ' '.join([word for word in essay if word not in stop_words])
                essays[i] = essay
            else:
                # If essay is null, set it to empty string
                essays[i] = ''

        print 'Done preprocessing in the {} step.'.format(fit_or_transform)
        return essays

    def getSATWords(self):
        with open('../data/SAT_words/1000_words.txt', 'r') as f:
            self.words_1000 = []
            for line in f:
                self.words_1000.append(line.strip())

        with open('../data/SAT_words/5000_words.txt', 'r') as f:
            self.words_5000 = []
            for line in f:
                self.words_5000.append(line.strip())

class FinalColumns(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        # final_cols = ['SAT_total_final', 'SAT_times_taken', 'High School GPA', 'Male', 'leader', 'arts', 'award', 'community', 'academic', 'gov', 'diversity', 'race_ecc', 'Home Country_US', '5000_words_frac']
        #
        # ethnicity_cols = [col for col in X.columns if col.find('Ethnicity_')>-1]
        # HS_perf_cols = [col for col in X.columns if col.find('HS_')>-1]
        # sports_cols = ['sportsVarsity', 'sportsCaptain']
        # essay_cols = ['essay_topic'+str(i) for i in range(1,8)]

        # final_cols.extend(ethnicity_cols)
        # final_cols.extend(HS_perf_cols)
        # final_cols.extend(sports_cols)
        # final_cols.extend(essay_cols)

        good_cols = ['SAT_total_final', 'SAT_times_taken', 'High School GPA', 'Male', 'leader', 'award', 'academic', 'gov', 'sportsVarsity', 'sportsCaptain', 'Ethnicity_Black', 'Ethnicity_White', 'HS_Steady']
        # good_cols.extend(essay_cols)

        X_model = X[good_cols].copy()
        print X_model.isnull().sum()

        lap = datetime.now()
        print 'Finished FinalColumns after {} seconds.'.format((lap-start).seconds)
        return X_model

class FinalColumnsCat(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        final_cols = ['Male', 'leader', 'award', 'academic', 'gov', 'sportsVarsity', 'sportsCaptain', 'Ethnicity_Black', 'Ethnicity_White', 'HS_Steady', 'SAT_total_final', 'SAT_times_taken', 'High School GPA']
        # '5000_words_frac', 'essay_topic1', 'essay_topic2', 'essay_topic3', 'essay_topic4', 'essay_topic5', 'essay_topic6', 'essay_topic7'
        print X[final_cols].isnull().sum()
        return X[final_cols]

class FinalColumnsNonCat(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        # print final_cols
        # print X[final_cols].isnull().sum()

        return X[final_cols]

def showModelResults(y_pred, y_test):
    print 'Accuracy:', accuracy_score(y_pred, y_test)
    print 'Precision:', precision_score(y_pred, y_test)
    print 'Recall:', recall_score(y_pred, y_test)
    print 'Confusion Matrix: ', confusion_matrix(y_pred, y_test)

if __name__=='__main__':
    df = pd.read_csv('../data/train.csv', low_memory=False)
    y = df.pop('top_school_final')
    df_train, df_valid, y_train, y_valid = train_test_split(df, y, train_size=0.7, random_state=123)
    df_train.reset_index(inplace=True)
    df_valid.reset_index(inplace=True)

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

    # params = {
    #     # 'model__C': np.logspace(0,3,5),
    #     'model__min_samples_split': range(2,3)
    #     }
    #
    # gs = GridSearchCV(pipeline, param_grid = params, cv=KFold(len(df_train),shuffle=True))
    # gs.fit(df_train, y_train)
    # y_pred = gs.predict(df_valid)

    # print 'Best parameters: {}'.format(gs.best_params_)
    # showModelResults(y_pred, y_valid)

    model = pipeline.fit(df_train, y_train)

    with open('../data/model.pkl', 'w') as f:
        pickle.dump(model, f)

    # THE VERY LAST THING TO DO (maybe create a new script for this)
    # df_test = pd.read_csv('../data/test.csv', low_memory=False)
    # y_test = df_test.pop('top_school_final')
    # best_model = LogisticRegression(C=100)
    # showModelResults(best_model, df, y, df_test, y_test)
