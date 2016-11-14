import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform, euclidean, cosine
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from sklearn.manifold import TSNE

class CleanEssays(object):
    def __init__(self):
        pass

    def updateEssayCols(self, df, old_cols):
        '''
        INPUT: df (DataFrame), old_cols (list)
        OUTPUT: None

        Updates the three essay columns to a simpler form, then drops the old columns.
        '''
        new_cols = ['essay_c'+str(i+1) for i,v in enumerate(old_cols)]
        for old,new in zip(old_cols,new_cols):
            df[new] = df[old].copy()
        df.drop(old_cols, axis=1, inplace=True)

    # Writing a function here bc we'll need to update these later again
    def updateWordCounts(self, df, essay_cols):
        '''
        INPUT: df (DataFrame), essay_cols (list)
        OUTPUT: None

        Creates/Updates columns that show the word count for the three essay columns.
        '''
        for col in essay_cols:
            df['wordcnt_'+col] = df[col].apply(lambda x: len(x.split()) if not x is np.nan and not x == None else x)

    def wordCountStats(self, df, cols, width=16, length=4):
        '''
        INPUT: df (DataFrame), cols (list)
        OUTPUT: summary (DataFrame), fig (matplotlib object)

        Gives a statistical summary of the specified columns, and plots a histogram for each of the columns.
        '''
        fig = plt.figure(figsize=(width,length))
        for i,col in enumerate(cols):
            ax = fig.add_subplot(1,len(cols),i+1)
            ax.boxplot(df[df[col].notnull()][col].values)
            ax.set_title(col)
        summary = df[cols].describe()
        return summary, fig

    def cleanEssayC3(self, essay):
        '''
        INPUT: essay (string)
        OUTPUT: cleaned essay (string)

        Receives an unformatted chunk of text, and extracts just the essay part.
        '''
        content = re.findall('Full Length Personal Statement([\s\S]*)', essay)
        if len(content)>0:
            cleaned = content[0].strip()
            brackets = re.findall('[[]\d+[]]', cleaned)
            if len(brackets)>0:
                pos = cleaned.find(brackets[0])
                return cleaned[:pos].strip()
        else:
            return np.nan

    def removeASCII(self, df, cols):
        '''
        INPUT: df (DataFrame), cols (list)
        OUTPUT: None

        Given a df and list of essay cols, this function removes ASCII characters in each entry in each col.
        '''
        for col in cols:
            df[col] = df[col].apply(lambda x: self._ASCII(x) if not x is np.nan and not x is None else x)

    def _ASCII(self, essay):
        '''
        Internal function for removeASCII function above
        '''
        # for exp in set(re.findall('\xe2\W*', essay)):
        for exp in set(re.findall('[^\w\s\d,.-]+', essay)):
            essay = essay.replace(exp, '')
        return essay

    def removeExtremes(self, df, cols):
        '''
        INPUT: df (DataFrame), cols (list)
        OUTPUT: None
        '''
        for col in cols:
            df[col] = df[col].apply(lambda x: x if not x is np.nan and not x is None and len(x.split())>200 and len(x.split())<1100 else np.nan)

    def checkOverlaps(self, df, cols):
        '''
        INPUT:
        OUTPUT:
        '''
        idx = []
        for col in cols:
            idx.append(df[df[col].notnull()==True].index)
        intersectc1c2 = np.intersect1d(idx[0],idx[1])
        intersectc1c3 = np.intersect1d(idx[0],idx[2])
        intersectc2c3 = np.intersect1d(idx[1],idx[2])
        print 'The num of rows that contain both c1 and c2: {}'.format(len(intersectc1c2))
        print 'The num of rows that contain both c1 and c3: {}'.format(len(intersectc1c3))
        print 'The num of rows that contain both c2 and c3: {}'.format(len(intersectc2c3))

    def removeOverlaps(self, df, cols, keep_col):
        '''
        INPUT: df (DataFrame), cols (list of cols), keep_col (string)
        OUTPUT: None
        '''
        _cols = cols[:]
        _cols.remove(keep_col)
        remove_col = _cols[0]
        idx = []
        for col in cols:
            idx.append(df[df[col].notnull()==True].index)
        intersect = np.intersect1d(idx[0],idx[1])
        df.ix[intersect, remove_col] = np.nan

    def consolidateEssays(self, df, cols):
        '''
        INPUT: df (DataFrame), cols (list of cols to consolidate)
        OUTPUT: essay text (string) or np.nan
        '''
        c1,c2,c3 = cols
        if type(df[c1])==str:
            return df[c1]
        elif type(df[c2])==str:
            return df[c2]
        elif type(df[c3])==str:
            return df[c3]
        else:
            return np.nan

    def removeDuplicates(self, df, col):
        '''
        INPUT: df (DataFrame), col (string)
        OUTPUT: None

        Drops rows where essay entries are duplicated (keeps one copy).
        '''
        duplicate_txt = []
        c = Counter(df[df[col].notnull()][col].values)
        for txt,num in c.most_common():
            if num>1:
                duplicate_txt.append(txt)
        for t in duplicate_txt:
            df.drop(df[df[col]==t][col].index[1:], axis=0, inplace=True)

class AnalyzeEssays(object):
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.stemmer = PorterStemmer()

    def stopWordsAndStem(self, essays):
        for i,essay in enumerate(essays):
            if not essay is np.nan and not essay is None:
                essay = re.sub('\xe2\W+', '', essay)
                essay = ' '.join([word for word in essay.split() if word not in self.stop_words])
                stemmed = []
                for word in essay.split():
                    try:
                        stemmed.append(self.stemmer.stem(word))
                    except UnicodeDecodeError:
                        pass
                essays[i] = ' '.join(stemmed)

    def removeASCII(self, essay):
        '''
        INPUT: essay (string)
        OUTPUT: modified essay (string)
        '''
        for exp in set(re.findall('\xe2\W+', essay)):
            essay = essay.replace(exp, '')
        return essay

    def removeStopWords(self, essay, custom_lst=None):
        '''
        INPUT: essay (string)
        OUTPUT: modified essay (string)
        '''
        lst = essay.split()
        output = ''
        for word in lst:
            if not custom_lst is None:
                if word.strip(punctuation) not in custom_lst and word.strip(punctuation) not in stopwords.words('english'):
                    output += word + ' '
            else:
                if word.strip(punctuation) not in stopwords.words('english'):
                    output += word + ' '
        return output

    def essayStemmer(self, essay, method, stem_or_lem = 'lem'):
        '''
        INPUT: essay (string), method (stem/lem function), stem_or_lem (boolean)
        OUTPUT: output (string)
        '''
        output = ''
        if stem_or_lem == 'stem':
            stemmer = method()
            for word in essay.strip().split():
                try:
                    output += stemmer.stem(word.strip(punctuation)) + ' '
                except UnicodeDecodeError:
                    pass
        elif stem_or_lem == 'lem':
            lemmer = method()
            for word in essay.strip().split():
                try:
                    output += lemmer.lemmatize(word.strip(punctuation)) + ' '
                except UnicodeDecodeError:
                    pass
        return output

    def dropNonWords(self, vocab, arr):
        '''
        INPUT: vocab (dict), arr (np.array)
        OUTPUT: arr_mod (np.array)

        Given a vocab dict, drops all 'non-words' (features that start with a non-letter) and returns a modified arr (with the dropped cols)
        '''
        remove_idx = []
        for word,idx in vocab.iteritems():
            if len(re.findall('[^a-z]', word)) > 0:
                remove_idx.append(idx)
        keep_idx = list(set(xrange(len(vocab))) - set(remove_idx))
        arr_mod = arr[:,keep_idx]
        return arr_mod

class dimReduction(object):
    def __init__(self):
        pass

    def getTSNE(self, model, components, mat):
        '''
        INPUT: model (NMF / PCA), mat (sparse matrix)
        OUTPUT: mat_tsne (reduced dim matrix)
        '''
        m = model(n_components=components)
        mat_reduced = m.fit_transform(mat.toarray())
        tsne = TSNE(n_components=2)
        mat_tsne = tsne.fit_transform(mat_reduced)
        return mat_tsne

class TopicModeling(object):
    def __init__(self):
        pass

    def plotOptimalNMF(self, X, max_components=6):
        '''
        INPUT: X (matrix)
        OUTPUT: None

        Plots n_components by reconstruction error
        '''
        errors = []
        components = range(2, max_components)
        for n in components:
            nmf = NMF(n_components = n)
            nmf.fit(X)
            errors.append(nmf.reconstruction_err_)
        plt.plot(components, errors)
        plt.xlabel('No of Components')
        plt.ylabel('Reconstruction Error')
        plt.title('NMF Reconstruction Error')

    def showTopWords(self, components, vec, no_words):
        topwords_idx = map(lambda x: np.argsort(x)[::-1][:no_words], components)
        topwords = map(lambda x: np.array(vec.get_feature_names())[x], topwords_idx)
        return topwords

    def similarEssaysNMF(self, essay, essays, topic_mat, vectorizer, dim_red_model, n=3):
        '''
        Returns a list of tuples of the most similar essays based on Euclidean distance of NMF scores.

        Tuple: (id, essay)
        '''
        # Perform Tfidf-Vectorization and NMF
        mat = vectorizer.transform([essay])
        mat_nmf = dim_red_model.transform(mat)

        # Calculate distances then return essays with least distance
        distances = []
        for row in topic_mat:
            distances.append(euclidean(row, mat_nmf))

        similar_idx = np.array(distances).argsort()[:n]
        similar_essays = essays[similar_idx]
        similar_essays = map(lambda x: unicode(x, 'utf-8'), similar_essays)

        return zip(similar_idx, similar_essays)

    def similarEssaysTfidf(self, essay, essays, sparse_mat, vectorizer, n=3):
        '''
        Returns a list of tuples of n most similar essays based on Cosine distance of tfidf scores.

        Tuple: (id, essay)
        '''
        # Perform Tfidf-Vectorization
        tfidf = vectorizer.transform([essay])
        tfidf = tfidf.toarray()

        # Calculate distances then return essays with least distance
        distances = []
        for row in sparse_mat:
            distances.append(cosine(row.toarray(), tfidf))

        similar_idx = np.array(distances).argsort()[1:n+1]
        similar_essays = essays[similar_idx]
        similar_essays = map(lambda x: unicode(x, 'utf-8'), similar_essays)

        return zip(similar_idx, similar_essays)

class ClusterTools(object):
    def __init__(self):
        pass

    def findBestK(self, mat, max_k=4):
        '''
        INPUT: mat (sparse matrix), max_k (max k to test)
        OUTPUT: None

        Tests for the best k using the silhouette score. Prints a message showing the best k and its silhouette score.
        '''
        self.silhouette_scores_ = []
        for k in xrange(2,max_k+1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(mat)
            score = silhouette_score(mat, kmeans.labels_)
            self.silhouette_scores_.append((score,k))
        self.best_score_ = sorted(self.silhouette_scores_, reverse=True)[0]
        print 'The best K is {}, with a silhouette score of {}.'.format(self.best_score_[1], self.best_score_[0])

    def topClusterWords(self, vec, mat, k=3, words=10):
        '''
        INPUT: vec (tfidf vectorizer), mat (sparse matrix), k (no of clusters), words (no of words to show)
        OUTPUT: top_words (list)
        '''
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(mat)
        centroids = kmeans.cluster_centers_
        top_args = map(lambda x: x.argsort()[-words:], centroids)
        vocab_inv = {v:k for k,v in vec.vocabulary_.iteritems()}
        top_words = []
        for cluster in top_args:
            top_words.append([vocab_inv[word] for word in cluster])
        return top_words

    def makeDendrogram(self, mat, df, n=100, dist_metric='cosine', cluster_method='complete'):
        '''
        INPUT:
        OUTPUT: None

        Makes a dendrogram given a set of specified parameters.
        '''
        subset_idx = np.random.randint(0, len(df), size=n)
        dist = pdist(mat.toarray()[subset_idx], metric=dist_metric)
        square = squareform(dist)
        Z = linkage(square, method=cluster_method)
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        _=dendrogram(Z, leaf_font_size=10,ax=ax)
        plt.ylabel('Height')
        plt.suptitle('Cluster Dendrogram ({})'.format(cluster_method.title()), fontweight='bold', fontsize=14)
