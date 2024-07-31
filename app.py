import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from movie_rec_sys import recommend_most_popular_per_genre, get_similar_movies

# Load the dataset
df = pd.read_csv('src/movies/datasets/movies.csv', low_memory=False)
df['GENRES'] = df['GENRES'].str.lower()

# Create combined features for similarity calculations
df['combined_features'] = df['ORIGINAL_TITLE'] + ' ' + df['GENRES'] + ' ' + df['DIRECTORS']
df['combined_features'] = df['combined_features'].str.replace('.', '', regex=False)
df['combined_features'] = df['combined_features'].str.replace(',', ' ', regex=False)
df['combined_features'] = df['combined_features'].str.lower()

filt_df = df[(df['VOTE_COUNT'] > 2000) & (df['TYPE'] == 'movie')]
filt_df = filt_df.reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words='english')

# df[df['GENRES'].isnull()] # bos yok

# TF-IDF Matrisinin olusturulmasi
tfidf_matrix = tfidf.fit_transform(filt_df['combined_features'])

# tfidf_matrix.shape
# tfidf.get_feature_names_out()

# Cosine Similarity Matrisinin Olusturulmasi
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Benzerliklere gore onerilerin yapilmasi
indices = pd.Series(filt_df.index, index=filt_df['ORIGINAL_TITLE'])


def get_similar_movies(tconst, cosine_sim=cosine_sim, df=filt_df, top_n=5):
    movie_index = df.index[df['TCONST'] == tconst].tolist()[0]
    if not isinstance(movie_index, int):
        raise ValueError("Movie index is not an integer.")
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i for i, _ in similarity_scores[1:top_n+1]]
    return df.iloc[movie_indices]







# Streamlit interface
st.title("Movie Recommendation System")

# Get most popular movies per genre
popular_movies_df = recommend_most_popular_per_genre(df)

# Select Movie
movie_choice = st.selectbox("Choose a movie:", popular_movies_df['ORIGINAL_TITLE'])

if st.button("Find Similar Movies", key="find_similar_button"):
    movie_tconst = popular_movies_df[popular_movies_df['ORIGINAL_TITLE'] == movie_choice]['TCONST'].values[0]
    similar_movies = get_similar_movies(movie_tconst, cosine_sim=cosine_sim, df=filt_df)
    st.write(f"Similar movies to {movie_choice}:")
    for movie in similar_movies['ORIGINAL_TITLE']:
        st.write(movie)


# Emotional Mood Buttons
st.title("How Do You Feel?")
if st.button("Happy", key="happy_button"):
    st.write("You feel happy! Here's a suggestion: 'The Grand Budapest Hotel'")
elif st.button("Sad", key="sad_button"):
    st.write("You feel sad! Here's a suggestion: 'The Pursuit of Happyness'")
elif st.button("Scared", key="scared_button"):
    st.write("You feel scared! Here's a suggestion: 'A Quiet Place'")
elif st.button("Excited", key="excited_button"):
    st.write("You feel excited! Here's a suggestion: 'Mad Max: Fury Road'")
elif st.button("Romantic", key="romantic_button"):
    st.write("You feel romantic! Here's a suggestion: 'Pride and Prejudice'")