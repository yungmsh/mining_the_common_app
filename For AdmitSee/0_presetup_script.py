import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import presetup
import seaborn as sns
import re
from collections import Counter

# Load data, update column names
df = pd.read_csv('../data/raw/raw_data.csv', low_memory=False)
ps = presetup.PreSetup()
col_dict = ps.parseCols('../data/reference/column_names.txt')
df.columns = ps.updateCols(df.columns.values)

# Keep students only
df = df[df['Who are you?']=='Admit Creating College / Grad School Profile'].copy()
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)

# Remove nulls
vals = ['YTowOnt9', 'ytowont9', 'czowOiIiOw==']
for v in vals:
    df.replace(to_replace=v, value=np.nan, inplace=True)

# Get School Acceptance Data
sc = presetup.Schools()
all_schools = sc.getSchools('../data/reference/table_references.csv')
all_schools = list(set(all_schools))
df_schools = pd.DataFrame(index=xrange(len(df)), columns=all_schools)
sc.extractFromApplied(df['Undergraduate Schools Applied'], df_schools)
df_schools = df_schools[all_schools]

# Top Schools
top_schools = ['Harvard University (Cambridge, MA)', 'Yale University (New Haven, CT)',
               'Cornell University (Ithaca, NY)', 'Columbia University (New York, NY)',
               'University of Pennsylvania (Philadelphia, PA)', 'Princeton University (Princeton, NJ)',
               'Brown University (Providence, RI)', 'Dartmouth College (Hanover, NH)',
               'Massachusetts Institute of Technology (Cambridge, MA)','Stanford University (Stanford, CA)']
df_topschools = df_schools[top_schools].copy()
for school in top_schools:
    df_topschools[school] = df_topschools[school].apply(lambda x: sc.cleanFromApplied(x) if type(x) == str else x)
df_topschools['any_top_school'] = (df_topschools.sum(axis=1)).apply(lambda x: 1 if x>0 else np.nan)

# Join df_topschools back with main df
df = df.join(df_topschools)

# Extract some more data from Attended col
df['any_top_school_v2'] =  df['Undergraduate Schools Attended'].apply(sc.extractFromAttended)
df['top_school_final'] = df.apply(sc.finalTopSchool, axis=1)

# Export
df.to_csv('../data/master.csv')
