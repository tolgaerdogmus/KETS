# IMPORTS #########################################################################################
import re
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# END IMPORTS ######################################################################################
# SETTINGS #########################################################################################
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
# END SETTINGS #####################################################################################

# Load the dataset
df = pd.read_csv('src/movies/datasets/movies_31-tem.csv', low_memory=False)

# GENERAL VARIABLES #########################################################################

# era categories for rec_by_era and streamlit components to use
ERA_CATEGORIES = {
    'Golden Age': (1900, 1949),
    'Classical Era': (1950, 1969),
    'New Hollywood': (1970, 1989),
    'Modern Era': (1990, 2009),
    'Contemporary Era': (2010, 2024)
}

MOOD_TO_GENRES = {
    'Huzurlu': (['comedy', 'adventure', 'family', 'animation'], []),
    'Duygusal': (['drama', 'romance', 'family'], []),
    'Hareketli': (['action', 'war', 'thriller', 'crime', 'adventure', 'western', 'sport'], []),
    'Karanlik': (['horror', 'thriller'], ['action', 'musical']),
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

# END GENERAL VARIABLES #####################################################################

# İÇERİK TEMELLİDE KULLANILACAK DATAFRAME ve TF-IDF İŞLEMLERİ #############################################
# Sadece tf-idf e sokulacak olan filtrelenecek veriseti oluştur. Diğer fonksiyonlar için normal df kullanılıyor.
filtre_df = df.copy()

# Filtrele
filtre_df = df[(df['VOTE_COUNT'] > 2000) & (df['AVG_RATING'] > 6.0)] # 2000 ve 6 için 13098 gözlem
# Filtrelendikten sonra hata vermemesi adına indexleri sıfırla
filtre_df = filtre_df.reset_index(drop=True)
#filtre_df.info() # TEST AMAÇLIDIR

# İngilizce gereksiz ve kendi başına anlam taşıyamayan kelimelerin dışlanması (örn: or, and, of, the..)
tfidf = TfidfVectorizer(stop_words='english')

# TF-IDF Matrisinin olusturulmasi
# ORIGINAL_TITLE + GENRES + DIRECTORS + OVERVIEW + YEAR = COMBINED_FEATURES
tfidf_matrix = tfidf.fit_transform(filtre_df['COMBINED_FEATURES'])


# Cosine Similarity Matrisinin Olusturulmasi
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Benzerliklere gore onerilerin yapilmasi
indices = pd.Series(filtre_df.index, index=filtre_df['ORIGINAL_TITLE'])



# FUNCTIONS ##########################################################################################

def rec_top_by_genre(df, genre='comedy', count=1, vote_threshold=500):
    if genre.lower() not in genre_categories:
        raise ValueError(f"Genre '{genre}' not found in the list of available genres.")

    genre_filter = df['GENRES'].str.contains(genre, case=False, na=False)
    vote_count_filter = df['VOTE_COUNT'] > vote_threshold
    filtered_df = df[genre_filter & vote_count_filter]

    top_movies = filtered_df.sort_values(by='AVG_RATING', ascending=False).head(count)
    top_recommendations_by_genre = top_movies[['TCONST', 'ORIGINAL_TITLE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']]

    return top_recommendations_by_genre




# print(rec_top_by_genre(df)) # TEST AMAÇLIDIR
# Example usage: Recommend top 10 popular movies for the genre 'Horror', vote count > 50k
# Ornek kullanim: en populer 10 adet Horror turu ve oy sayisi 50binin uzerinde
# rec_top_by_genre(df, 'horror', count= 10, vote_threshold=50000)


def rec_top_all_genres(df=df):
    # Create an empty list to store recommendations
    # Tavsiye icin bos bir liste tanimla
    recommendations = []

    # Get all unique genres
    # Tum virgul ile ayrilmis genreleri tek tek al
    all_genres = set(genre for sublist in df['GENRES'].dropna().str.split(',') for genre in sublist)

    for genre in all_genres:
        # Filter the dataset by the selected genre
        # genre basina veri setini filtrele
        genre_filter = df['GENRES'].str.contains(genre, case=False, na=False)
        filtered_df = df[genre_filter]

        if not filtered_df.empty:
            # Get the most popular movie for this genre
            # Genre icin en vote_countu ve avg_rating i yuksekleri diz ve birinci elemani al
            most_popular = filtered_df.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=False).iloc[0]
            recommendations.append(most_popular)

    # Create a DataFrame for the recommendations
    # Data frame e cevir
    recommendations_df = pd.DataFrame(recommendations)

    # Ensure the DataFrame has the required columns
    # Gereken kolonlarin olup olmadigini kontrol et
    if not recommendations_df.empty:
        # Select relevant columns to display, ensuring all columns exist
        # Gosterilecek kolonlari sec
        columns_to_display = ['TCONST', 'ORIGINAL_TITLE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']
        recommendations_df = recommendations_df[[
            col for col in columns_to_display if col in recommendations_df.columns
        ]]

    return recommendations_df


# Kullanim:
#print(rec_top_all_genres(df))  # TEST AMAÇLIDIR

def rec_top_by_genre(df=df, genre_category='Comedy', count=1, vote_threshold=500):
    # Ensure the genre category is lowercase
    genre_category = genre_category.lower()

    if genre_category not in genre_categories:
        raise ValueError(f"Genre category '{genre_category}' not found.")

    # Create a filter for the selected genre and vote count
    genre_filter = df['GENRES'].str.contains(genre_category, case=False, na=False)
    vote_count_filter = df['VOTE_COUNT'] > vote_threshold
    filtered_df = df[genre_filter & vote_count_filter]

    # Sort the filtered dataset by average rating in descending order and select the top x count
    top_movies = filtered_df.sort_values(by='AVG_RATING', ascending=False).head(count)

    # Select relevant columns to display
    top_recommendations_by_genre = top_movies[['TCONST', 'ORIGINAL_TITLE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']]

    return top_recommendations_by_genre


def rec_most_popular(df=df, count= 1):
    # Veri setini VOTE_COUNT ve AVG_RATING'e göre sıralayarak en popüler içeriği bul
    most_popular = df.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=[False, False]).head(count)

    # Gereken kolonları seç
    columns_to_display = ['TCONST', 'ORIGINAL_TITLE', 'TYPE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']
    most_popular = most_popular[columns_to_display]

    return most_popular

# print(rec_most_popular(df))  # TEST AMAÇLIDIR

#TCONST a göre içerik tabanlı getir
def get_similar_by_id(tconst, cosine_sim=cosine_sim, df=filtre_df, count=10):
    movie_index = df.index[df['TCONST'] == tconst].tolist()[0]
    if not isinstance(movie_index, int):
        raise ValueError("Movie index is not an integer.")
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i for i, _ in similarity_scores[1:count+1]]
    return df.iloc[movie_indices]


def get_similar_by_title(title, cosine_sim=cosine_sim, df=filtre_df, count=10):
    # Compile a case-insensitive regex pattern for the given title
    pattern = re.compile(re.escape(title), re.IGNORECASE)

    # Find the index of the first title that matches the pattern
    movie_index = df[df['ORIGINAL_TITLE'].str.contains(pattern)].index[0]

    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i for i, _ in similarity_scores[1:count + 1]]
    return df.iloc[movie_indices]

# deneme_sim = get_similar_by_title('itanic', count=10)  # TEST AMAÇLIDIR

def follow_your_mood(df, mood='Huzurlu', count=10):
    """
    Ruh haline uygun filmleri öneren fonksiyon.

    Parametreler:
    df (pd.DataFrame): Filmler hakkında bilgileri içeren DataFrame.
    mood (str): Ruh hali.
    count (int): Getirilecek film sayısı.

    Döndürülen:
    pd.DataFrame: Önerilen filmler DataFrame'i.
    """
    if mood not in MOOD_TO_GENRES:
        st.write("Seni Ruhsuz!")
        return pd.DataFrame()  # Return an empty DataFrame if mood is not found

    selected_genres, excluded_genres = MOOD_TO_GENRES[mood]
    selected_genres = set(selected_genres)
    excluded_genres = set(excluded_genres)

    def genre_match(genres):
        genre_list = set(map(str.lower, map(str.strip, genres.split(','))))
        return bool(selected_genres.intersection(genre_list)) and not bool(excluded_genres.intersection(genre_list))

    filtered_df = df[df['GENRES'].apply(genre_match)]

    if filtered_df.empty:
        st.write(f"{mood} ruh haline uygun film bulunamadı.")
        return pd.DataFrame()  # Return an empty DataFrame if no movies are found

    movies_for_mood = filtered_df.nlargest(count, ['VOTE_COUNT', 'AVG_RATING'])

    return movies_for_mood

#print(follow_your_mood(mood='Karanlik')) # TEST AMAÇLIDIR


def rec_random_movies(df=filtre_df, count=10):
    # Ensure the num_movies does not exceed the number of rows in the DataFrame
    num_movies = min(count, len(df))

    # Use the sample method to get random rows
    random_movies = df.sample(n=count)  # ,random_state=1 Set random_state for reproducibility

    # Sort the random movies by VOTE_COUNT and AVG_RATING
    sorted_movies = random_movies.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=[False, False])

    return sorted_movies

# random_movies = rec_random_movies(filtre_df, 10) # TEST AMAÇLIDIR
# print(type(random_movies)) # TEST AMAÇLIDIR
def rec_top_directors(df=df, dir_count=5, movie_count=5):
    """
    En çok izlenen beş yönetmene göre rastgele 5 film önerisi yapan fonksiyon.

    Parametreler:
    df (pd.DataFrame): Filmler hakkında bilgileri içeren DataFrame.
    dir_count (int): En çok izlenen yönetmen sayısı.
    movie_count (int): Her yönetmenden seçilecek film sayısı.

    Döndürülen:
    pd.DataFrame: Önerilen filmler DataFrame'i.
    """
    # Yönetmen başına toplam izlenme (oy) sayısını hesapla ve en çok izlenenleri seç
    top_directors = df.groupby('DIRECTORS')['VOTE_COUNT'].sum().nlargest(dir_count).index

    recommendations = []

    for director in top_directors:
        # Yönetmen bazında filtreleme ve rastgele n film seç
        director_movies = df[df['DIRECTORS'] == director].sample(n=min(movie_count, df[df['DIRECTORS'] == director].shape[0]))

        # Sadece gerekli kolonları seç
        director_movies = director_movies[['TCONST', 'DIRECTORS', 'ORIGINAL_TITLE', 'AVG_RATING']]

        # Yönetmen ve filmleri ekle
        for _, movie in director_movies.iterrows():
            recommendations.append(movie.to_dict())

    return pd.DataFrame(recommendations)


# En çok izlenen beş 5 yönetmene göre  rastgele 5 film. Bunuda degistirebiliriz.
# print(rec_top_directors()) # TEST AMAÇLIDIR

def rec_by_era(df=df, start_year=1900, end_year=1949, count=10):
    """
    Belirtilen döneme göre film önerisi yapan fonksiyon.

    Parametreler:
    df (pd.DataFrame): Filmler hakkında bilgileri içeren DataFrame.
    start_year (int): Başlangıç yılı.
    end_year (int): Bitiş yılı.
    count (int): Getirilecek film sayısı.

    Döndürülen:
    pd.DataFrame: Önerilen filmler DataFrame'i.
    """
    # Ensure 'YEAR' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['YEAR']):
        df['YEAR'] = pd.to_datetime(df['YEAR'], errors='coerce')

    # Create datetime objects for start and end years
    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year, month=12, day=31)

    # Filter movies by the specified year range
    era_movies = df[(df['YEAR'] >= start_date) & (df['YEAR'] <= end_date)]

    # Sort movies by 'VOTE_COUNT' and 'AVG_RATING'
    sorted_movies = era_movies.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=[False, False])

    # Select top movies
    top_movies = sorted_movies[['TCONST', 'DIRECTORS', 'ORIGINAL_TITLE', 'AVG_RATING']].head(count)

    return top_movies

rec_by_era(df)






# END FUNCTIONS ##########################################################################################

# STREAMLIT FUNCTIONS ##########################################################################################

def display_mood_selection(df):
    with st.expander('Ruh haline göre tavsiyeler:', expanded=True):
        # Create a dropdown list with a default item
        mood_options = ['Hangi moda girmek istersin?'] + list(MOOD_DISPLAY_NAMES.values())
        selected_mood_display_name = st.selectbox('Select Mood:', mood_options)

        # Convert selected display name back to internal mood key
        internal_mood = next((key for key, value in MOOD_DISPLAY_NAMES.items() if value == selected_mood_display_name), None)

        if internal_mood and internal_mood != 'Hangi moda girmek istersin?':
            recommendations = follow_your_mood(df, mood=internal_mood)
            if not recommendations.empty:
                st.write(internal_mood + " **için sonuçlar:**")
                st.dataframe(recommendations)
            else:
                st.write("No recommendations available.")

def display_genre_selection(df):
    with st.expander("Türlere göre tavsiyeler:", expanded=False):
        genre_selection = st.selectbox("Select Genre", options=[g.capitalize() for g in genre_categories])

        if st.button("Get Recommendations"):
            try:
                top_movies = rec_top_by_genre(df, genre_category=genre_selection, count=10)
                if top_movies.empty:
                    st.write(f"No movies found for genre '{genre_selection}'.")
                else:
                    st.write("**Top Movies:**")
                    st.dataframe(top_movies)
            except ValueError as e:
                st.write(e)

def select_movie_era(df):
    with st.expander("Zaman aralıklarına göre tavsiyeler:", expanded=False):
        # Create a dropdown list with a default item
        era_list = ['Hangi döneme yolculuk yapalım?'] + list(ERA_CATEGORIES.keys())
        selected_era = st.selectbox('Select a movie era:', era_list)

        # Only proceed if a valid era is selected
        if selected_era != 'Hangi döneme yolculuk yapalım?':
            start_year, end_year = ERA_CATEGORIES[selected_era]
            recommendations = rec_by_era(df, start_year, end_year)
            if not recommendations.empty:
                st.write("**Recommendations for the Era:**")
                st.dataframe(recommendations)
            else:
                st.write("No recommendations available.")


def find_similar_movies(filtre_df):
    with st.expander("Popülerlere benzer tavsiyeler:"):
        popular_movies_df = rec_top_all_genres(filtre_df)
        movie_choice = st.selectbox("Film:", popular_movies_df['ORIGINAL_TITLE'])

        if st.button("Buna benzer bul"):
            movie_tconst = popular_movies_df[popular_movies_df['ORIGINAL_TITLE'] == movie_choice]['TCONST'].values[0]
            similar_movies = get_similar_by_id(movie_tconst)
            st.write(f"**{movie_choice} benzeri filmler:**")
            for movie in similar_movies['ORIGINAL_TITLE']:
                st.write(movie)

def main():
    # Adding Image to web app
    st.set_page_config(page_title="K.E.T.S. Kişisel Eğlence Tavsiye Sistemi", page_icon='Images/Cookie_Cat.png')

    # CSS to center the image
    st.markdown(
        """
        <style>
        .centered-image img {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Define the URL or path to your image
    banner_img_url = 'Images/Cookie_Cat.png'

    # Add the image with a custom class for centering
    st.markdown(
        f"""
        <div class="centered-image">
            <img src="{banner_img_url}" width="400" alt="K.E.T.S. Sunar!">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("K.E.T.S. Movie Recommendation System")

    display_mood_selection(df)
    select_movie_era(df)
    display_genre_selection(df)
    find_similar_movies(filtre_df)

if __name__ == "__main__":
    main()