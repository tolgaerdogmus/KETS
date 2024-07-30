import pandas as pd
import numpy as np
# pip install IMDbPY
from imdb import IMDb


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

imdb_ = IMDb()

people = imdb_.search_person('Mel Gibson')
for person in people:
    print(person.personID, person['name'])


#miuul's movie dataset
df_mi_movie = pd.read_csv('src/movies/datasets/miuul_movies.csv', low_memory=False)
df_mi_movie.info()
df_mi_movie = df_mi_movie.rename(columns={
    'imdb_id': 'TCONST'
})


#merge director and writer with movies_ratings
df_crew = pd.read_csv('src/movies/datasets/title.crew.tsv', sep='\t')
df_crew.columns = df_crew.columns.str.upper()
df_movies_ratings = pd.read_csv('src/movies/datasets/movies_ratings.csv')
df_movies_ratings_directors = pd.merge(df_movies_ratings, df_crew, on='TCONST', how='inner')
df_movies_ratings_directors.head()
df_artists = pd.read_csv('src/movies/datasets/artists.csv')
df_movies_ratings_directors['NCONST'] = df_movies_ratings_directors['DIRECTORS']
df_movies_ratings_directors['DIRECTORS'] = df_movies_ratings_directors['PRIMARYNAME']
df_movies_ratings_directors = pd.merge(df_movies_ratings_directors, df_artists, on='NCONST', how='inner')
df_movies_ratings_directors.info()

# replace NCONST numbers in WRITERS with names from artists.csv
df_exploded_mov_rat_dir = df_movies_ratings_directors.explode('WRITERS')
df_mov_rat_dir_wri = pd.merge(df_exploded_mov_rat_dir, df_artists, left_on='WRITERS', right_on='NCONST', how='left')
df_mov_rat_dir_wri['WRITERS'] = df_exploded_mov_rat_dir['PRIMARYNAME']
df_exploded_mov_rat_dir_grouped = df_mov_rat_dir_wri.groupby(df_mov_rat_dir_wri.index)['WRITERS'].apply(list)
df_mov_rat_dir_wri['WRITERS'] = df_exploded_mov_rat_dir_grouped


df_movie_final = df_mov_rat_dir_wri[['TCONST_x', 'ORIGINALTITLE', 'TITLETYPE', 'AVERAGERATING', 'NUMVOTES', 'GENRES', 'DIRECTORS', 'ISADULT', 'STARTYEAR', 'RUNTIMEMINUTES']].rename(columns={
    'TCONST_x': 'TCONST',
    'ORIGINALTITLE': 'ORIGINAL_TITLE',
    'TITLETYPE': 'TYPE',
    'AVERAGERATING': 'RATING',
    'NUMVOTES': 'VOTE_COUNT',
    'GENRES': 'GENRES',
    'DIRECTORS': 'DIRECTORS',
    'ISADULT': 'IS_ADULT',
    'STARTYEAR': 'YEAR',
    'RUNTIMEMINUTES': 'TIME_MINUTES'
})


df_movie_final = pd.merge(df_movie_final, df_mi_movie, on='TCONST', how='left')
df_movie_final.info()

df_movie_final = df_movie_final[['TCONST',
                                 'ORIGINAL_TITLE',
                                 'TYPE',
                                 'RATING',
                                 'VOTE_COUNT',
                                 'popularity',
                                 'GENRES',
                                 'overview',
                                 'DIRECTORS',
                                 'IS_ADULT',
                                 'YEAR',
                                 'TIME_MINUTES',
                                 'poster_path',
                                 'video']].rename(columns={
    'overview': 'OVERVIEW',
    'popularity': 'POPULARITY',
    'poster_path': 'POSTER_PATH',
    'RATING': 'AVG_RATING',
    'video': 'VIDEO'

})
df_movie_final.info()
df_movie_final.drop(columns=['VIDEO'], inplace=True)
df_movie_final.to_csv('src/movies/datasets/movies.csv')

original_title_to_find = 'Titanic'


# Find the row where ORIGINALTITLE matches the value
result = df_movie_final.loc[df_movie_final['ORIGINAL_TITLE'] == original_title_to_find, 'OVERVIEW']

# Print the TCONST value
if not result.empty:
    print("TCONST:", result.values[0])
else:
    print("Not found")
# TEMPORARY FILTERING CODE
# Ratings 5 and more
# Vote count 100 and more

df_movie_final.replace({'\\N': np.nan}, inplace=True)
df_movie_final['YEAR'] = pd.to_datetime(df_movie_final['YEAR'])
df_movie_final['YEAR'] = df_movie_final['YEAR'].dt.strftime('%d/%m/%Y')

df_movie_final.to_csv('movie_rating_director.csv', index=False)

df_movie_final.info()
df_movie_final.sort_values(by='RATING', ascending=False)

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






