import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


#
name = '/Users/avokado/PycharmProjects/miuul/Final/name.basics.tsv'
df1 = pd.read_csv(name, sep='\t')
df1_expanded = df1.assign(knownForTitles=df1['knownForTitles'].str.split(',')).explode('knownForTitles')
df1.head()


#
rat ='/Users/avokado/PycharmProjects/miuul/Final/title.ratings.tsv'
df2 = pd.read_csv(rat, sep='\t')
df2.head()


#
basic ='/Users/avokado/PycharmProjects/miuul/Final/title.basics.tsv'
df3 = pd.read_csv(basic , sep='\t')
df3.head()


# İki veri seti (name+rat)
merged_df = pd.merge(df1_expanded, df2 , left_on='knownForTitles', right_on='tconst', how='inner')
merged_df.head()

#3.veri setini ekleme (name_rat_basic)
final_merged_df = pd.merge(merged_df, df3, on='tconst', how='inner')
final_merged_df.head()

def check_detail(dataframe):
    d = {'SHAPE': dataframe.shape,
         'COLUMNS': dataframe.columns,
         'INDEX': dataframe.index,
         'VALUE TYPES': dataframe.dtypes,
         'DUPLICATED VALUES': dataframe.duplicated().sum(),
         'NUMBER OF UNIQUE VALUES': dataframe.nunique(),
         'ANY MISSING VALUES': dataframe.isnull().values.any(),
         'MISSING VALUES': dataframe.isnull().sum(),
         'DESCRIBE.T': dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T}
    hashtags = '---------------------------'
    for key, val in d.items():
        print(f'{hashtags} {key} {hashtags}')
        print(val)
    print(f'{hashtags} {"LIST END"} {hashtags}')

check_detail(dataframe=final_merged_df)

# id yi basa alma ve siralama
cols = final_merged_df.columns.tolist()
cols.insert(0, cols.pop(cols.index('tconst')))
df = final_merged_df[cols]

df = df.sort_values(by='tconst').reset_index(drop=True)
df.head()

#Dönüsümler
# Yılları ve runtimeMinutes'ı sayısal
df['birthYear'] = pd.to_numeric(df['birthYear'], errors='coerce')
df['deathYear'] = pd.to_numeric(df['deathYear'], errors='coerce')
df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')
df['endYear'] = pd.to_numeric(df['endYear'], errors='coerce')
df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')

# `isAdult` kolonunu sayısal (int)
df['isAdult'] = df['isAdult'].astype(int)

# Id'ler string
df['nconst'] = df['nconst'].astype(str)
df['tconst'] = df['tconst'].astype(str)

# Verinin son durumunu kontrol edin
print(df.dtypes)
df.isnull().values.any()
check_detail(dataframe=df)

df['startYear'].isnull()