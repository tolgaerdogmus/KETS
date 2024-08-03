# IMPORTS #########################################################################################
import streamlit as st
import pandas as pd
import re
from streamlit_lottie import st_lottie
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Set page config at the very beginning
st.set_page_config(page_title="K.E.T.S. Movie Recommender", page_icon='🍿', layout="wide")

# era categories for rec_by_era and streamlit components to use
ERA_CATEGORIES = {
    'Golden Age': (1900, 1949),
    'Classical Era': (1950, 1969),
    'New Hollywood': (1970, 1989),
    'Modern Era': (1990, 2009),
    'Contemporary Era': (2010, 2024)
}

MOOD_TO_GENRES = {
    'Huzurlu': (['comedy', 'adventure', 'family', 'animation'], ['action', 'war', 'horror', 'thriller', 'crime']),
    'Duygusal': (['drama', 'romance', 'family'], []),
    'Hareketli': (['action', 'war', 'thriller', 'crime', 'adventure', 'western', 'sport'], ['music', 'musical']),
    'Karanlik': (['horror', 'thriller'], ['action', 'music', 'family', 'comedy', 'romance']),
    'Gizemli': (['mystery', 'crime', 'thriller'], []),
    'Geek': (['sci-fi', 'fantasy', 'animation'], []),
    'Dans': (['musical', 'music'], []),
    'Cocuk': (['animation', 'family', 'musical'], ['action', 'adult', 'war', 'horror', 'thriller', 'crime', 'western']),
    'Entel': (['biography', 'history', 'documentary', 'film-noir', 'short'], [])
}

# Mapping for display names
MOOD_DISPLAY_NAMES = {
    'Huzurlu': 'Huzurlu 😊',
    'Duygusal': 'Duygusal 😢',
    'Hareketli': 'Hareketli ⚡',
    'Karanlik': 'Karanlik 🌑',
    'Gizemli': 'Gizemli 🕵️',
    'Geek': 'Geek 🤓',
    'Dans': 'Dans 💃',
    'Cocuk': 'Kids 🧒',
    'Entel': 'Intellectual 📚'
}

genre_categories = [
    'action', 'adventure', 'sci-fi', 'fantasy',
    'animation', 'family', 'comedy',
    'biography', 'history', 'documentary',
    'crime', 'mystery', 'thriller',
    'horror', 'musical', 'music',
    'news', 'reality-tv', 'romance',
    'sport', 'war', 'western'
]

# END GENERAL VARIABLES ##################################################################


# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('src/movies/datasets/movies_31-tem.csv', low_memory=False)
    return df

df = load_data()

# TF-IDF and Cosine Similarity calculation
@st.cache_resource
def calculate_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['COMBINED_FEATURES'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_similarity(df)
# Include all the functions you provided (rec_top_by_genre, rec_top_all_genres, rec_most_popular,
# get_similar_by_id, get_similar_by_title, follow_your_mood, rec_random_movies, rec_top_directors, rec_by_era)
def rec_top_by_genre(df=df, genre='comedy', count=1, vote_threshold=500):
    if genre.lower() not in genre_categories:
        raise ValueError(f"Genre '{genre}' not found in the list of available genres.")

    genre_filter = df['GENRES'].str.contains(genre, case=False, na=False)
    vote_count_filter = df['VOTE_COUNT'] > vote_threshold
    filtered_df = df[genre_filter & vote_count_filter]

    top_movies = filtered_df.sort_values(by='AVG_RATING', ascending=False).head(count)
    top_recommendations_by_genre = top_movies[['TCONST', 'ORIGINAL_TITLE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']]

    return top_recommendations_by_genre

def rec_top_all_genres(df=df):
    recommendations = []
    all_genres = set(genre for sublist in df['GENRES'].dropna().str.split(',') for genre in sublist)

    for genre in all_genres:
        genre_filter = df['GENRES'].str.contains(genre, case=False, na=False)
        filtered_df = df[genre_filter]

        if not filtered_df.empty:
            most_popular = filtered_df.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=False).iloc[0]
            recommendations.append(most_popular)

    recommendations_df = pd.DataFrame(recommendations)

    if not recommendations_df.empty:
        columns_to_display = ['TCONST', 'ORIGINAL_TITLE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']
        recommendations_df = recommendations_df[[
            col for col in columns_to_display if col in recommendations_df.columns
        ]]

    return recommendations_df

def rec_most_popular(df=df, count=1):
    most_popular = df.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=[False, False]).head(count)
    columns_to_display = ['TCONST', 'ORIGINAL_TITLE', 'TYPE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']
    most_popular = most_popular[columns_to_display]
    return most_popular

def get_similar_by_id(tconst, cosine_sim=cosine_sim, df=df, count=10):
    movie_index = df.index[df['TCONST'] == tconst].tolist()[0]
    if not isinstance(movie_index, int):
        raise ValueError("Movie index is not an integer.")
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i for i, _ in similarity_scores[1:count+1]]
    return df.iloc[movie_indices]

def get_similar_by_title(title, cosine_sim=cosine_sim, df=df, count=10):
    pattern = re.compile(re.escape(title), re.IGNORECASE)
    movie_index = df[df['ORIGINAL_TITLE'].str.contains(pattern)].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i for i, _ in similarity_scores[1:count + 1]]
    return df.iloc[movie_indices]

def follow_your_mood(df, mood='Huzurlu', count=10):
    if mood not in MOOD_TO_GENRES:
        st.write("Seni Ruhsuz!")
        return pd.DataFrame()

    selected_genres, excluded_genres = MOOD_TO_GENRES[mood]
    selected_genres = set(selected_genres)
    excluded_genres = set(excluded_genres)

    def genre_match(genres):
        genre_list = set(map(str.lower, map(str.strip, genres.split(','))))
        return bool(selected_genres.intersection(genre_list)) and not bool(excluded_genres.intersection(genre_list))

    filtered_df = df[df['GENRES'].apply(genre_match)]

    if filtered_df.empty:
        st.write(f"{mood} ruh haline uygun film bulunamadı.")
        return pd.DataFrame()

    movies_for_mood = filtered_df.nlargest(count, ['VOTE_COUNT', 'AVG_RATING'])
    return movies_for_mood

def rec_random_movies(df=df, count=10):
    num_movies = min(count, len(df))
    random_movies = df.sample(n=count)
    sorted_movies = random_movies.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=[False, False])
    return sorted_movies

def rec_top_directors(df=df, dir_count=5, movie_count=5):
    if 'DIRECTORS' not in df.columns or df['DIRECTORS'].isna().all():
        st.warning("Directors information is not available.")
        return pd.DataFrame()

    top_directors = df.groupby('DIRECTORS')['VOTE_COUNT'].sum().nlargest(dir_count).index
    recommendations = []

    for director in top_directors:
        director_movies = df[df['DIRECTORS'] == director].sample(n=min(movie_count, df[df['DIRECTORS'] == director].shape[0]))
        director_movies = director_movies[['TCONST', 'DIRECTORS', 'ORIGINAL_TITLE', 'AVG_RATING']]
        for _, movie in director_movies.iterrows():
            recommendations.append(movie.to_dict())

    return pd.DataFrame(recommendations)

def rec_by_era(df=df, start_year=1900, end_year=1949, count=10):
    if not pd.api.types.is_datetime64_any_dtype(df['YEAR']):
        df['YEAR'] = pd.to_datetime(df['YEAR'], errors='coerce')

    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year, month=12, day=31)

    era_movies = df[(df['YEAR'] >= start_date) & (df['YEAR'] <= end_date)]
    sorted_movies = era_movies.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=[False, False])
    top_movies = sorted_movies[['TCONST', 'DIRECTORS', 'ORIGINAL_TITLE', 'AVG_RATING']].head(count)
    return top_movies



## STREAMLIT ##############################################################
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def display_movie_list(movies, section_key):
    for index, movie in movies.iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{movie['ORIGINAL_TITLE']}**")
            st.write(f"Rating: {movie['AVG_RATING']:.1f}")
            if 'DIRECTORS' in movie and pd.notna(movie['DIRECTORS']):
                st.write(f"Director(s): {movie['DIRECTORS']}")
        with col2:
            # Check if we've already found similar movies for this one
            if f"similar_{movie['TCONST']}" not in st.session_state:
                if st.button(f"Find Similar", key=f"{section_key}_{index}"):
                    similar_movies = get_similar_by_id(movie['TCONST'], cosine_sim=cosine_sim, df=df)
                    st.session_state[f"similar_{movie['TCONST']}"] = similar_movies
                    st.session_state.selected_movie = movie['ORIGINAL_TITLE']
            else:
                st.write("Similar movies found")
        st.markdown("---")



def show_main_page():
    st.title("🎬 K.E.T.S. Kişisel Eğlence Tavsiye Sistemi")
    st.markdown("Discover your next favorite movie based on your mood!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🌈 Mood-based Movie Recommendations")
        mood_options = ['Choose your mood'] + list(MOOD_DISPLAY_NAMES.values())
        selected_mood_display_name = st.selectbox('How are you feeling today?', mood_options, key='mood_selector')

        internal_mood = next((key for key, value in MOOD_DISPLAY_NAMES.items() if value == selected_mood_display_name), None)

        if internal_mood and internal_mood != 'Choose your mood':
            if st.button("Get Recommendations"):
                recommendations = follow_your_mood(df, mood=internal_mood)
                if not recommendations.empty:
                    st.subheader(f"Top picks for {selected_mood_display_name} mood:")
                    display_movie_list(recommendations, 'mood')
                else:
                    st.info("No recommendations available for this mood. Try another one!")

    with col2:
        lottie_mood = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_khzniaya.json")
        st_lottie(lottie_mood, key="mood_animation")

    # Additional features in expanders
    with st.expander("Explore More Features"):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Genre Explorer", "Top Rated Movies", "Era-based Recommendations", "Director Recommendations", "Random Picks"])

        with tab1:
            st.header("🎭 Genre Explorer")
            selected_genre = st.selectbox("Select a genre:", options=['Select a genre'] + genre_categories)

            if selected_genre != 'Select a genre':
                if st.button("Get Genre Recommendations"):
                    top_genre_movies = rec_top_by_genre(df, genre=selected_genre, count=10)
                    st.subheader(f"Top {selected_genre} Movies:")
                    display_movie_list(top_genre_movies, 'genre')

        with tab2:
            st.header("🌟 Top Rated Movies")
            if st.button("Get Top Rated Movies"):
                top_movies = rec_most_popular(df, count=10)
                display_movie_list(top_movies, 'top_rated')

        with tab3:
            st.header("🕰️ Era-based Recommendations")
            era_list = ['Select an era'] + list(ERA_CATEGORIES.keys())
            selected_era = st.selectbox('Choose an era:', era_list)

            if selected_era != 'Select an era':
                if st.button("Get Era Recommendations"):
                    start_year, end_year = ERA_CATEGORIES[selected_era]
                    era_movies = rec_by_era(df, start_year, end_year)
                    st.subheader(f"Top picks from {selected_era} ({start_year}-{end_year}):")
                    display_movie_list(era_movies, 'era')

        with tab4:
            st.header("🎬 Director Recommendations")
            if st.button("Get Director Recommendations"):
                director_movies = rec_top_directors(df)
                st.subheader("Top movies from popular directors:")
                display_movie_list(director_movies, 'director')

        with tab5:
            st.header("🎲 Random Movie Picks")
            if st.button("Get Random Movies"):
                random_movies = rec_random_movies(df, count=10)
                st.subheader("Discover something new:")
                display_movie_list(random_movies, 'random')


def show_similar_movies_page(tconst):
    movie_data = df[df['TCONST'] == tconst]
    if not movie_data.empty:
        movie_title = movie_data['ORIGINAL_TITLE'].values[0]
        st.title(f"Movies similar to {movie_title}")

        similar_movies = get_similar_by_id(tconst, cosine_sim, df)
        display_movie_list(similar_movies, f'similar_{tconst}')

        if st.button("Back to Main Page"):
            st.session_state.page = 'main'
            st.experimental_rerun()
    else:
        st.warning(f"Movie data not found for TCONST: {tconst}")
        if st.button("Back to Main Page"):
            st.session_state.page = 'main'
            st.experimental_rerun()


def display_movie_list(movies, section_key):
    for index, movie in movies.iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{movie['ORIGINAL_TITLE']}**")
            st.write(f"Rating: {movie['AVG_RATING']:.1f}")
            if 'DIRECTORS' in movie and pd.notna(movie['DIRECTORS']):
                st.write(f"Director(s): {movie['DIRECTORS']}")
        with col2:
            similar_movies_url = f"?page=similar&tconst={movie['TCONST']}"
            st.markdown(f'<a href="{similar_movies_url}" target="_blank">Find Similar</a>', unsafe_allow_html=True)
        st.markdown("---")


def show_similar_movies_page(tconst):
    movie_data = df[df['TCONST'] == tconst]
    if not movie_data.empty:
        movie_title = movie_data['ORIGINAL_TITLE'].values[0]
        st.title(f"Movies similar to {movie_title}")

        similar_movies = get_similar_by_id(tconst, cosine_sim, df)
        display_movie_list(similar_movies, f'similar_{tconst}')
    else:
        st.warning(f"Movie data not found for TCONST: {tconst}")


def main():
    # Get query parameters
    query_params = st.query_params

    # Check if we're on the similar movies page
    if query_params.get("page") == "similar" and "tconst" in query_params:
        show_similar_movies_page(query_params["tconst"])
    else:
        show_main_page()


if __name__ == "__main__":
    main()