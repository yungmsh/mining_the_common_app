import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from collections import Counter, OrderedDict
import re
import time
import string
import en

class CleanEssays(object):
    def getEssayIndices(self, X):
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

        # lap = datetime.now()
        # print 'Finished CleanEssays after {} seconds.'.format((lap-start).seconds)
        idx = X[X['essay_final'].notnull()].index
        return (idx, X)

        # return X

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

class AnalyzeEssays(object):
    def fit(self, X):
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

    def transform(self, X):
        print len(X)
        essays = self.preprocess(X, 'transform')
        print len(essays)
        mat = self.vec.transform(essays)
        mat_nmf = self.nmf.transform(mat)
        print len(X)
        df_nmf = pd.DataFrame(mat_nmf, columns = self.essay_topics)
        X = X.join(df_nmf)

        # Impute missing values with 'avg' value for each topic
        for col,value in zip(self.essay_topics, self.avg_topics):
            X[col].fillna(value=value, inplace=True)

        # Impute missing values for SAT word cols
        X['1000_words_cnt'].fillna(value=self.words_1000_mean, inplace=True)
        X['5000_words_frac'].fillna(value=self.words_5000_mean, inplace=True)

        print X.columns
        return X

    def preprocess(self, X, fit_or_transform):
        essays = X['essay_final'].values
        punct = string.punctuation
        stop_words = stopwords.words('english')
        wn = WordNetLemmatizer()

        for df_idx,essay_tuple in zip(X.index.values, enumerate(essays)):
            essay = essay_tuple[1]
            essay_idx = essay_tuple[0]
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
                for j,word in enumerate(essay):
                    try:
                        essay[j] = wn.lemmatize(en.verb.present(word.split(punct)))
                    except:
                        pass
                # Extract data for SAT words
                c = Counter(essay)
                count_1000 = sum([c[word] for word in self.words_1000])
                count_5000 = sum([c[word] for word in self.words_5000])
                X.loc[df_idx, '1000_words_cnt'] = count_1000
                if len(essay)==0:
                    X.loc[df_idx,'5000_words_frac'] = 0
                else:
                    X.loc[df_idx, '5000_words_frac'] = count_5000 / float(len(essay))

                # Remove stop words
                essay = ' '.join([word for word in essay if word not in stop_words])
                essays[essay_idx] = essay
            else:
                # If essay is null, set it to empty string
                essays[essay_idx] = ''

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
