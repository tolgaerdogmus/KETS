import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# pip install IMDbPY
#from imdb import IMDb

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv('src/movies/datasets/movies.csv', low_memory=False)
df.info()
df.head()
df = df[df['GENRES'].notna() & (df['GENRES'] != '')]
df = df[df['OVERVIEW'].notna() & (df['OVERVIEW'] != '')]
df = df[df['AVG_RATING'].notna() & df['TYPE'].notna()]
# TF-IDF vektörizer oluştur ## Bu Vektörize ediyor.
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

film_genre_matrix = tfidf_vectorizer.fit_transform(df['OVERVIEW'])

def recommend_movies(title, df, film_genre_matrix, min_rating=5.0):
    # Filter the DataFrame to include only movies with high average ratings
    movie_df = df[(df['TYPE'].str.lower() == 'movie') & (df['AVG_RATING'] >= min_rating)]

    # Get the index of the movie that matches the title
    idx = movie_df.index[movie_df['ORIGINAL_TITLE'] == title].tolist()

    if not idx:
        return "Movie not found in the dataset or does not meet the rating criteria."

    idx = idx[0]

    # Calculate the cosine similarity between the movie's overview and all other movies in the filtered movie dataframe
    cosine_sim = cosine_similarity(film_genre_matrix[idx], film_genre_matrix[movie_df.index]).flatten()

    # Get the indices of the most similar movies, excluding the movie itself
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar movies

    movie_indices = [movie_df.index[i[0]] for i in sim_scores]

    return df[['ORIGINAL_TITLE', 'GENRES', 'AVG_RATING']].iloc[movie_indices]

# Example usage
recommended_movies = recommend_movies('The Exorcist', df, film_genre_matrix)
print(recommended_movies)
