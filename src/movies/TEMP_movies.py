import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



# TEMPORARY FILTERING CODE
# Ratings 5 and more
# Vote count 100 and more

#
df_ratings = pd.read_csv('src/movies/datasets/title.ratings.tsv', sep='\t')
df_ratings.head()
df_ratings['numVotes'].describe().T

rating_temp = df_ratings.loc[df_ratings['numVotes'].values > 100]

rating_temp.shape # x > 30 - 668391 | x > 100 - 365611

rating_temp2 = rating_temp.loc[rating_temp['averageRating'] > 5.000]

rating_temp2.shape  # 325493
#
df_movies = pd.read_csv('src/movies/datasets/title.basics.tsv', sep='\t')
df_movies.head()

df_movies_ratings = pd.merge(rating_temp2, df_movies, on='tconst', how='inner')

df_movies_ratings.shape

df_movies_ratings.columns = df_movies_ratings.columns.str.upper()

df_movies_ratings.head()
df_movies_ratings.to_csv('src/movies/datasets/movies_ratings.csv', index=False)
#
df_artists = pd.read_csv('src/movies/datasets/name.basics.tsv', sep='\t')
#df_artists_expanded = df_artists.assign(knownForTitles=df_artists['knownForTitles'].str.split(',')).explode('knownForTitles')
df_artists.columns = df_artists.columns.str.upper()
# Explode the KNOWNFORTITLES column in the first dataset
df_artists_exploded = df_artists.assign(TCONST=df_artists['KNOWNFORTITLES'].str.split(',')).explode('TCONST')

# Filter the exploded DataFrame to keep only the rows with TCONST values present in the second dataset
df_filtered_artists = df_artists_exploded[df_artists_exploded['TCONST'].isin(df_movies_ratings['TCONST'])]

# Group by NCONST and aggregate the KNOWNFORTITLES and TCONST values into lists
df_artists_final = df_filtered_artists.groupby(['NCONST', 'PRIMARYNAME', 'BIRTHYEAR', 'DEATHYEAR', 'PRIMARYPROFESSION']).agg({
    'KNOWNFORTITLES': lambda x: list(set(','.join(x).split(','))),
    'TCONST': lambda x: list(x)
}).reset_index()


df_artists_final = df_artists_final.drop('KNOWNFORTITLES', axis=1)

df_artists_final.to_csv('src/movies/datasets/artists.csv', index=False)





