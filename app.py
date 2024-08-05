# IMPORTS #########################################################################################
import streamlit as st
import pandas as pd
import re
import base64
from io import BytesIO
from PIL import Image
from streamlit_lottie import st_lottie
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import configparser
import os

# Set page config at the very beginning
st.set_page_config(page_title="KETS", page_icon='🍿', layout="wide") # wide
# Accessing a single secret
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
    st.success("TMDB API key successfully loaded from secrets!")
except KeyError:
    st.error("TMDB API key not found in secrets. Please check your configuration.")
    st.stop()


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to config.ini
config_path = os.path.join(script_dir, 'config.ini')

# Load configuration
config = configparser.ConfigParser()

# Check if the config file exists
# if os.path.exists(config_path):
#     config.read(config_path)
#     TMDB_API_KEY = config['API']['TMDB_API_KEY']
# else:
#     st.error("Configuration file not found. Please ensure config.ini is in the same directory as the script.")
#     TMDB_API_KEY = None


# era categories for rec_by_era and streamlit components to use
ERA_CATEGORIES = {
    'Golden Age (1900 - 1949)': (1900, 1949),
    'Classical (1950 - 1969)': (1950, 1969),
    'New Hollywood (1970 - 1989)': (1970, 1989),
    'Modern (1990 - 2009)': (1990, 2009),
    'Contemporary (2010 - Today)': (2010, 2024)
}

MOOD_TO_GENRES = {
    'Huzurlu': (['comedy', 'adventure', 'family', 'animation'], ['action', 'war', 'horror', 'thriller', 'crime', 'drama']),
    'Duygusal': (['drama', 'romance', 'family'], ['action','horror', 'western', 'thriller']),
    'Hareketli': (['action', 'sci-fi', 'war', 'thriller', 'crime', 'adventure', 'western', 'sport'], ['music', 'musical']),
    'Karanlik': (['horror', 'thriller'], ['action', 'music', 'family', 'comedy', 'romance']),
    'Gizemli': (['mystery', 'crime', 'thriller', 'sci-fi'], ['action', 'western', 'sport', 'family', 'comedy']),
    'Geek': (['sci-fi', 'fantasy', 'animation'], ['family', 'music', 'musical', 'comedy', 'romance']),
    'Dans': (['musical', 'music'], ['family', 'short', 'biography']),
    'Cocuk': (['animation', 'family', 'musical'], ['action', 'adult', 'war', 'horror', 'thriller', 'crime', 'western']),
    'Entel': (['biography', 'history', 'documentary', 'film-noir', 'news', 'short'], ['comedy','romance', 'horror', 'drama', 'crime', 'adventure'])
}

# Mapping for display names

MOOD_DISPLAY_NAMES = {
    'Huzurlu': 'Happy 😊',
    'Duygusal': 'Touchy-feely 😢',
    'Hareketli': 'High tension ⚡',
    'Karanlik': 'Darkness 💀',
    'Gizemli': 'Mysterious 🕵️',
    'Geek': 'Geek 🤓',
    'Dans': 'Dance 💃',
    'Cocuk': 'Kids 🧒',
    'Entel': 'Brainiac 🧠'
}

genre_categories = [
    'action', 'adventure', 'sci-fi', 'fantasy',
    'animation', 'family', 'comedy',
    'biography','film-noir', 'history', 'documentary',
    'crime', 'mystery', 'thriller',
    'horror', 'musical', 'music',
    'news', 'romance',
    'sport', 'war', 'western'
]

# END GENERAL VARIABLES ##################################################################


# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('src/movies/datasets/movies_31-tem.csv', low_memory=False)
    df = df[(df['VOTE_COUNT'] > 2500) & (df['AVG_RATING'] > 6.0)]
    df = df.reset_index(drop=True)
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

def follow_your_mood(df=df, mood='Huzurlu', count=10):
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

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def add_bg_from_url(url):
    st.markdown(
         f"""
         <style>
         .stApp {{
            background-image: url({url});
             background-attachment: fixed;
             background-size: auto auto;  # Use the image's original size
             background-repeat: no-repeat;
             background-position: center bottom;  # Center horizontally, align to bottom
             position: fixed;
             top: 0;
             left: 0;
             right: 0;
             bottom: 0;
             z-index: -1;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Example usage
add_bg_from_local('Images/nebulabg.png')  # Use this for local images
# add_bg_from_url('https://example.com/your-image.jpg')  # Use this for images hosted online


def add_footer():
    footer = """
        <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #0E1117;
                color: #FAFAFA;
                text-align: center;
                padding: 10px;
                font-size: 14px;
            }
            .footer a {
            color: #9CA3AF;  /* Subtle gray color */
            text-decoration: none;
            transition: color 0.3s ease;
            }
            .footer a:hover {
                color: #D1D5DB;  /* Lighter gray on hover for better interactivity */
            }
        </style>
    <div class="footer">
        KETS - Kişisel Eğlence Tavsiye Sistemi - Developed with ❤️ by 
        <a href="https://www.linkedin.com/in/gncgulce/" target="_blank"> | Gülce Kästel 🦉🧠🎈 </a>
        <a href="https://www.linkedin.com/in/zeynep-bakan-ba1996308/" target="_blank">| Zeynep Bakan 👩🏻‍🏫🧠🎓 </a>
        <a href="https://www.linkedin.com/in/tolgaerdogmus/" target="_blank">| Tolga Erdoğmuş 👽🐱🛠️</a>       
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)


def load_logo():
    return Image.open('Images/kets_kedi.png')


def show_main_page():
    # Custom CSS for unified logo and title
    st.markdown("""
    <style>
    .unified-header {
        display: flex;
        align-items: center;
        gap: 20px;  /* Space between logo and title */
    }
    .unified-header img {
        width: 100px;  /* Adjust as needed */
    }
    .title-container {
        display: flex;
        flex-direction: column;
    }
    .title-container h1 {
        margin: 0;
        font-size: 2.5em;  /* Adjust as needed */
    }
    .title-container p {
        margin: 0;
        font-size: 1em;  /* Adjust as needed */
    }
    </style>
    """, unsafe_allow_html=True)

    # Load the logo
    logo = load_logo()

    # Convert PIL Image to base64
    buffered = BytesIO()
    logo.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Create unified header with logo and title
    unified_header = f"""
    <div class="unified-header">
        <img src="data:image/png;base64,{img_str}" alt="KETS Logo">
        <div class="title-container">
            <h1>KETS</h1>
            <p>Endless choice, limitless joy!</p>
        </div>
    </div>
    """
    st.markdown(unified_header, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🌡️ Vibe-o-Meter!")
        mood_options = ['Catch your mood'] + list(MOOD_DISPLAY_NAMES.values())
        selected_mood_display_name = st.selectbox('Select your mood', mood_options, key='mood_selector', label_visibility="collapsed", index=0)

        internal_mood = next((key for key, value in MOOD_DISPLAY_NAMES.items() if value == selected_mood_display_name),
                             None)

        if internal_mood and internal_mood != 'Catch your mood':
            if st.button("Get Recommendations"):
                recommendations = follow_your_mood(df, mood=internal_mood)
                if not recommendations.empty:
                    st.subheader(f"Top picks for {selected_mood_display_name} mood:")
                    display_movie_list(recommendations, 'mood')
                else:
                    st.info("No recommendations available for this mood. Try another one!")

    with col2:
        lottie_mood = load_lottieurl("https://lottie.host/ca07f33a-da6a-45d1-b225-5b2c3a326df4/M8Nh7RCNEk.json")
        st_lottie(lottie_mood, key="mood_animation")

    # Additional features in expanders
    with st.expander("🆘 Can't find right vibe?"):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Genre Explorer", "Top Rated", "Time Machine", "Favorite Directors", "Random Picks"])

        with tab1:
            st.header("🎭 Genre Explorer")
            selected_genre = st.selectbox("Choose below:", options=['Select a genre'] + genre_categories, label_visibility="collapsed", index=0)

            if selected_genre != 'Select a genre':
                if st.button("Get Genre Recommendations"):
                    top_genre_movies = rec_top_by_genre(df, genre=selected_genre, count=10)
                    st.subheader(f"Top {selected_genre} Movies:")
                    display_movie_list(top_genre_movies, 'genre')

        with tab2:
            st.header("🌟 Top Rated")
            if st.button("Get Top Rated Movies"):
                top_movies = rec_most_popular(df, count=10)
                display_movie_list(top_movies, 'top_rated')

        with tab3:
            st.header("🕰️ Time Machine")
            era_list = ['Select a generation'] + list(ERA_CATEGORIES.keys())
            selected_era = st.selectbox('Go to:', era_list)

            if selected_era != 'Select a generation':
                if st.button("Get Era Recommendations"):
                    start_year, end_year = ERA_CATEGORIES[selected_era]
                    era_movies = rec_by_era(df, start_year, end_year)
                    st.subheader(f"Top picks from {selected_era} ({start_year}-{end_year}):")
                    display_movie_list(era_movies, 'era')

        with tab4:
            st.header("🎬 Favorite Directors")
            if st.button("Get Director Recommendations"):
                director_movies = rec_top_directors(df)
                st.subheader("Top movies from popular directors:")
                display_movie_list(director_movies, 'director')

        with tab5:
            st.header("🎲 Random Picks")
            if st.button("Try your luck"):
                random_movies = rec_random_movies(df, count=10)
                st.subheader("Discover something new:")
                display_movie_list(random_movies, 'random')

    add_footer()


def show_similar_movies_page(tconst):
    movie_data = df[df['TCONST'] == tconst]
    if not movie_data.empty:
        movie_title = movie_data['ORIGINAL_TITLE'].values[0]
        st.title(f"Movies similar to {movie_title}")

        similar_movies = get_similar_by_id(tconst, cosine_sim, df)
        display_movie_list(similar_movies, f'similar_{tconst}')

        if st.button("Back to Main Page"):
            st.query_params.clear()
            st.experimental_rerun()
    else:
        st.warning(f"Movie data not found for TCONST: {tconst}")
        if st.button("Back to Main Page"):
            st.query_params.clear()
            st.experimental_rerun()

    add_footer()


@st.cache_data
def get_poster_url(imdb_id):
    if not TMDB_API_KEY:
        st.error("TMDB API key not found. Please check your configuration.")
        return None
    try:
        search_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={TMDB_API_KEY}&external_source=imdb_id"
        response = requests.get(search_url)
        data = response.json()

        if 'movie_results' in data and data['movie_results']:
            tmdb_id = data['movie_results'][0]['id']
            details_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
            response = requests.get(details_url)
            data = response.json()

            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        st.error(f"Error fetching poster for IMDb ID {imdb_id}: {str(e)}")
    return None

def display_movie_list(movies, section_key):
    for index, movie in movies.iterrows():
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            poster_url = get_poster_url(movie['TCONST'])
            if poster_url:
                st.image(poster_url, width=100)
            else:
                st.write("No poster available")

        with col2:
            st.write(f"**{movie['ORIGINAL_TITLE']}**")
            st.write(f"Rating: {movie['AVG_RATING']:.1f}")
            if 'DIRECTORS' in movie and pd.notna(movie['DIRECTORS']):
                st.write(f"Director(s): {movie['DIRECTORS']}")

        with col3:
            similar_movies_url = f"?page=similar&tconst={movie['TCONST']}"
            st.markdown(f'<a href="{similar_movies_url}" target="_blank">Find Similar</a>', unsafe_allow_html=True)

        st.markdown("---")


def main():
    try:
    # Get query parameters
        query_params = st.query_params

    # Check if we're on the similar movies page
        if query_params.get("page") == "similar" and "tconst" in query_params:
            show_similar_movies_page(query_params["tconst"])
        else:
            show_main_page()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(traceback.format_exc())


if __name__ == "__main__":
    main()