import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

def histogramSAT(df, year=None, before_after = 'before', lim_min=100, lim_max=2400):
    '''
    Plots a histogram showing the SAT scores of those students who graduated before a specified year.
    '''
    if not year==None:
        if before_after=='before':
            mask_year = df['Undergraduate Graduation Year']<year
        else:
            mask_year = df['Undergraduate Graduation Year']>year
        df[(mask_year)]['Highest Composite SAT Score'].hist(range=(lim_min,lim_max))
    else:
        df['Highest Composite SAT Score'].hist(range=(lim_min,lim_max))

def parseSAT(x):
    '''
    Helper function to parse SAT scores. Used inside a Pandas 'apply' function on the df['Highest SAT Scores'] series.
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

def finalizeSAT(df):
    # Change NaNs to None in SAT_total_temp so we can apply a max function
    df['SAT_total_temp'] = df['SAT_total_temp'].apply(lambda x: None if str(x)=='nan' else x)
    # Take max of 2 columns
    df['SAT_total_final'] = np.max(df[['Highest Composite SAT Score','SAT_total_temp']], axis=1)
    # Remove faulty entries that don't end in '0' (SAT scores are multiples of 10)
    df['SAT_total_final'].apply(lambda x: None if x%10>0 else x)

def parseEthnicity(x):
    '''
    Helper function to parse Ethnicity ethnicity. Used inside a Pandas 'apply' function on the df['Ethnicity'] series.
    '''
    x = x.lower().split('|')
    if 'prefer not to share' in x:
        del x[x.index('prefer not to share')]
    if len(x) == 0:
        return np.nan
    else:
        return x

def showTopPhrases(df, col, n=50):
    '''
    Helper function to show top phrases in the df['High School Extracurricular Activities'] series
    '''
    c = Counter()
    for x in df[df[col].notnull()][col]:
        data = re.findall('[[]\S+[]]\s[[]\S+[]]\s(.+)', x)
        c.update(data)
    return c.most_common()[:n]

def parseECC(x, lst):
    '''
    Helper function to parse Extracurricular activities. Used inside a Pandas 'apply' function on the df['High School Extracurricular Activities'] series.
    '''
    if type(x)==str:
        x = x.lower()
        for word in lst:
            if x.find(word)>-1:
                return 1
        return 0
    else:
        return 0

def getAllSports(x):
    '''
    Helper function to get all sports data. Used inside a Pandas apply function on the df['High School Sports Played'] series.
    '''
    if not x in (None,np.nan):
        data = re.findall('[[]\S+[]]\s[[]\S+[]]\s(.+)', x)
        sports = []
        for d in data:
            try:
                int(d)
            except ValueError:
                if d[:2] != 'No' and d[:3] != 'Yes':
                    sports.append(d)
        return sports
    else:
        return []

def getUniqueSports(all_sports):
    '''
    Helper function to get all unique sports.
    '''
    all_sports2 = []
    map(lambda x: all_sports2.extend(x), all_sports)
    return list(set(all_sports2))

def makeSportsDummies(df, unique_sports):
    '''
    Creates dummy variables for each sport category, initialized with 0s.
    '''
    for sport in unique_sports:
        df['sports_'+sport] = 0

def parseSports(df, unique_sports):
    '''
    Fills in the dummy variables for each sport category.
    '''
    for i in df.index:
        raw_data = df.loc[i,'High School Sports Played']
        if not raw_data is np.nan:
            for sport in unique_sports:
                if sport in raw_data:
                    df.loc[i,'sports_'+sport] = 1

def parseVarsity(x, unique_sports, regexp):
    '''
    Parses through sports text to determine whether someone participated in a varsity sport.
    '''
    if not x is None and not x is np.nan:
        data = regexp.findall(x)
        if len(data)>3 and data[2] == 'Yes':
            return 1
        else:
            return 0
    else:
        return 0

def parseCaptain(x, unique_sports, regexp):
    '''
    Parses through sports text to determine whether someone was a captain of a sport.
    '''
    if not x is None and not x is np.nan:
        data = regexp.findall(x)
        if len(data)>4 and data[3] == 'Yes':
            return 1
        else:
            return 0
    else:
        return 0

def showNulls(df, col):
    '''
    Shows the percentage of each column that contains null values.
    '''
    return df.isnull().sum(axis=0) / len(df) * 100

def exploreCategorical(df, x, y, i, j):
    '''
    Explore rates of 'y' for each unique value in a particular variable

    INPUTS:
        df (dataframe): Pandas DataFrame object
        x (str): input variable
        y (str): output variable
        i (int): no of rows in plot grid
        j(int): no of cols in plot grid
    OUTPUT:
        Prints the rate of top_school_final for each unique value in col
        Plots histograms of the same data to visualize.

    Note: Works better with categorical data (or data with a limited no of unique values)
    '''
    fig = plt.figure(figsize=(15,6))
    for idx,val in enumerate(df[x].unique()):
        if not str(val) == 'nan':
            print val
            a = df[df[x]==val][y].value_counts()
            print a/a.sum()
            fig.add_subplot(i,j,idx+1)
            plt.hist(df[df[x]==val][y])
            plt.title(str(val))
