import re

import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 100)

df_main = pd.read_csv('src/movies/datasets/movies.csv', low_memory=False) # DtypeWarning kapatmak icin
df_metadata = pd.read_csv('src/movies/datasets/movies_metadata.csv', low_memory=False) # DtypeWarning kapatmak icin
df_metadata.rename(columns={'imdb_id': 'TCONST', 'overview': 'OVERVIEW'}, inplace=True)
df_metadata.info()

columns_to_merge = ['TCONST', 'OVERVIEW']
df_selected_cols = df_metadata[columns_to_merge]

df_merged = pd.merge(df_main, df_selected_cols, on='TCONST', how='left')

df_merged.info()

df_merged['TYPE'].unique()
video_movies = df_merged[df_merged['TYPE'] == 'movie']
df_cleaned = df_merged.dropna(subset=['OVERVIEW'])
df_cleaned.info()
df_cleaned['TYPE'].unique()
movies = df_cleaned[df_cleaned['TYPE'] == 'movie']
movies['TYPE'].unique()
df_cleaned = df_cleaned[df_cleaned['TYPE'] == 'movie']

df_year_temp = pd.DataFrame
df_cleaned['YEAR'] = pd.to_datetime(df_cleaned['YEAR'])

# Extract the year part from the YEAR column (assuming YEAR column exists and is in a proper date format)


df_cleaned['COMBINED_FEATURES'] = (df_cleaned['ORIGINAL_TITLE'] + ' '
                                   + df_cleaned['GENRES'] + ' '
                                   + df_cleaned['DIRECTORS'] + ' '
                                   + df_cleaned['OVERVIEW'] + ' '
                                   + df_cleaned['YEAR'].astype(str))
print(df_cleaned['COMBINED_FEATURES'].sample(1).to_string())
# Clean column by removing all non-letter and non-digit characters
df_cleaned['COMBINED_FEATURES'] = df_cleaned['COMBINED_FEATURES'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))


df_cleaned['COMBINED_FEATURES'] = df_cleaned['COMBINED_FEATURES'].str.lower()
df_cleaned.info()


# Drop columns
df_cleaned.drop(columns=['YEAR_INT'], inplace=True)

# Reset index
df_3107 = df_cleaned.reset_index(drop=True)

df_3107.info()

df_cleaned.to_csv('src/movies/datasets/movies_31-tem.csv', index=False)

