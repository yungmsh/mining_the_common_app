import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import re

class PreSetup(object):
    def __init__(self):
        pass

    def parseCols(self, filename):
        '''
        INPUT: path to file (str)
        OUTPUT: dict {key: id, val: (profile_type, data_type, column_name)}

        This function takes a filename and returns a dictionary where key is the field id and value isa tuple containing info about the field.
        '''
        self.d = {}
        with open(filename, 'r') as f:
            for line in f:
                data = line.split()
                profile_type = data.pop()
                data_type = data.pop()
                self.d[data[0]] = (profile_type, data_type, ' '.join(data[1:]))
        return self.d

    def updateCols(self, raw_cols):
        '''
        INPUT: raw_cols (list)
        OUTPUT: col_names (list)

        Given a list of raw column names (which contain field ids), this function returns a list of interpretable column names.
        '''
        col_ids = []
        for c in raw_cols:
            col_ids.append(c.split('_')[-1])
        col_names = []
        for c in col_ids:
            try:
                col_names.append(self.d[c][-1])
            except KeyError:
                col_names.append(c)
        return col_names

class Schools(object):
    def __init__(self):
        pass

    def getSchools(self, filename):
        '''
        INPUT: filename (str)
        OUTPUT: list

        This function takes a filename and returns a list of schools.
        '''
        df = pd.read_csv(filename)
        df = df[df['channel_id']==2]
        return df['title'].values

    def extractFromApplied(self, arr, new_df):
        '''
        INPUT: arr (list), new_df (DataFrame)
        OUTPUT: None

        This function takes a numpy array and updates the acceptance values in the specified df.
        '''
        for row,x in enumerate(arr):
            if not x is np.nan:
                data = re.findall('[[]\S+[]]\s[[]\S+[]]\s(.+)', x)
                for i,v in enumerate(data):
                    if i>0 and (v.lower()=='enrolled' or v.lower()=='accepted' or v.lower()=='denied' or v.lower()== 'withdrew' \
                    or v.lower()=='dont remember' or v.lower()=='accepted from waitlist' or v.lower()== 'waitlisted' \
                    or v.lower()=='no result'):
                    # We choose to manually write out these conditions rather check for 'element in list' for computational gains
                        new_df.ix[row,data[i-1]] = v

    def cleanFromApplied(self, x):
        '''
        INPUT: x (string)
        OUTPUT: binary int

        This function takes a string as input and returns 1 if accepted/enrolled, 0 if not.
        '''
        x = x.lower()
        if x.find('accepted')>-1 or x.find('enrolled')>-1:
            return 1
        else:
            return 0

    def extractFromAttended(self, x):
        '''
        INPUT: x (string or np.nan)
        OUTPUT: binary int

        This function takes in some string/nan input and returns 1 if its user attended top school, 0 if not.
        '''
        if not x is np.nan:
            schools = re.findall('[[]\S+[]]\s[[]\S+[]]\s(.+)', x)
            if len(top_schools) == len(set(top_schools + schools)):
                return 1
            else:
                return 0

    def finalTopSchool(self, x):
        '''
        INPUT: x (string or np.nan)
        OUTPUT: binary int

        This function takes in some string/nan input and returns 1 if either one of two columns has a 1, 0 if not.
        '''
        if x['any_top_school'] is np.nan or x['any_top_school_v2'] is np.nan:
            return np.nan
        else:
            if x['any_top_school']==1 or x['any_top_school_v2']==1:
                return 1
            else:
                return 0

top_schools = ['Harvard University (Cambridge, MA)', 'Yale University (New Haven, CT)',
               'Cornell University (Ithaca, NY)', 'Columbia University (New York, NY)',
               'University of Pennsylvania (Philadelphia, PA)', 'Princeton University (Princeton, NJ)',
               'Brown University (Providence, RI)', 'Dartmouth College (Hanover, NH)',
               'Massachusetts Institute of Technology (Cambridge, MA)','Stanford University (Stanford, CA)']

if __name__ == '__main__':
    df = pd.read_csv('../data/raw_data.csv', low_memory=False)
    df.drop('entry_id', axis=1, inplace=True)
    ps = PreSetup()
    col_dict = ps.parseCols('../data/column_names.txt') # grab column names
    df.columns = ps.updateCols(df.columns.values) # update column names

    # We're only interested in the profiles of those who have gone through the college admissions process and are not school faculty or parents.
    df = df[df['Who are you?']=='Admit Creating College / Grad School Profile'].copy()
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    # Remove all base64 nulls
    vals = ['YTowOnt9', 'ytowont9', 'czowOiIiOw==']
    for v in vals:
        df.replace(to_replace=v, value=np.nan, inplace=True)

    sc = Schools()
    all_schools = sc.getSchools('../data/table_references.csv') # get school names
    all_schools = list(set(all_schools)) # remove duplicates

    # Set up a separate df for just school acceptance data
    df_schools = pd.DataFrame(index=xrange(len(df)), columns=all_schools)
    sc.extractFromApplied(df['Undergraduate Schools Applied'], df_schools)

    df_schools = df_schools[all_schools] # only keep school cols

    # Set up a separate df for just top schools
    df_topschools = df_schools[top_schools].copy()
    # Extract acceptance info for each 'top school'
    for school in top_schools:
        df_topschools[school] = df_topschools[school].apply(lambda x: sc.cleanFromApplied(x) if type(x) == str else x)
    # Create new binary col: acceptance into ANY of the top schools
    df_topschools['any_top_school'] = (df_topschools.sum(axis=1)).apply(lambda x: 1 if x>0 else np.nan)

    df = df.join(df_topschools) # join back to original df

    # Extract data from the 'attended' column
    df['any_top_school_v2'] =  df['Undergraduate Schools Attended'].apply(sc.extractFromAttended)

    # Combine data from to columns to get a final column
    df['top_school_final'] = df.apply(sc.finalTopSchool, axis=1)

    # Remove rows that are too sparse
    df = df[df['Internal Use - Calculated Undergrad Price']>4]

    # Train test split
    df_train, df_test = train_test_split(df, train_size=0.7)
    df_train.to_csv('../data/train.csv')
    df_test.to_csv('../data/test.csv')
    df.to_csv('../data/master.csv')
