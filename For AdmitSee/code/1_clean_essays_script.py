import pandas as pd
import numpy as np
import re
import essay_analysis as ea
import presetup
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

#####################
# Start cleaning essay cols

old_cols = ['Undergraduate Personal Statement', 'Undergraduate Essay Details', 'NEW Personal Statement']
ce = ea.CleanEssays()
ce.updateEssayCols(df, old_cols)

essay_cols = ['essay_c1','essay_c2','essay_c3']
ce.updateWordCounts(df,essay_cols)
wordcnt_cols = ['wordcnt_'+x for x in essay_cols]

# Clean essay_c3 first
df['essay_c3_edit'] = df['essay_c3'].apply(lambda x: ce.cleanEssayC3(x) if not x is np.nan else x)

# Then deal with c1 and c2
for old,new in zip(essay_cols, ['essay_c1_edit', 'essay_c2_edit']):
    df[new] = df.apply(lambda x: x[old] if x['Undergraduate Personal Statement Type'] == 'Full Length Personal Statement' else np.nan, axis=1)

# Remove ASCII
essay_cols_edit = ['essay_c1_edit', 'essay_c2_edit', 'essay_c3_edit']
# ce.removeASCII(df, essay_cols_edit)

# Remove extremes
ce.removeExtremes(df, essay_cols_edit)
# ce.updateWordCounts(df, essay_cols_edit)
wordcnt_cols_edit = [x+'_edit' for x in wordcnt_cols]

# Remove overlaps
ce.removeOverlaps(df, ['essay_c1_edit','essay_c2_edit'], keep_col = 'essay_c1_edit')

# Store good essays into 'essay_final'
df['essay_final'] = df.apply(lambda x: ce.consolidateEssays(x, essay_cols_edit), axis=1)
ce.removeDuplicates(df, 'essay_final')
# ce.updateWordCounts(df, ['essay_final'])

# Export
df.to_csv('../data/cleaned_data.csv')

print 'Finished cleaning all essays.\nNew file created/updated: cleaned_data.csv'
