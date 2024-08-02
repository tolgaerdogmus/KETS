# IMPORTS #########################################################################################
import re
import streamlit as st
import pandas as pd
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

def rec_top_by_genre(df=df, genre='comedy', count=1, vote_threshold = 500):
    # Filter the dataset by the selected genre
    # GENRE ve VOTE_COUNT gore filtrele

    genre_filter = df['GENRES'].str.contains(genre, case=False, na=False)
    vote_count_filter = df['VOTE_COUNT'] > vote_threshold
    filtered_df = df[genre_filter & vote_count_filter]

    # Sort the filtered dataset by average rating in descending order and select the top x count
    # Filtrelenmis datasetini AVG_RATING e gore sirala ve en bastan x tane tane getir
    top_movies = filtered_df.sort_values(by='AVG_RATING', ascending=False).head(count)

    # Select relevant columns to display
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

def follow_your_mood(df=df, mood='Huzurlu', count=10):
    # Karanlik[0] = istenilen turler Karanlik[1] = haric tutulacaklar
    mood_to_genres = {
        'Huzurlu': (['comedy', 'adventure', 'family', 'animation'], []),
        'Duygusal': (['drama', 'romance', 'family'], []),
        'Hareketli': (['action', 'war', 'thriller', 'crime', 'adventure', 'western', 'sport'], []),
        'Karanlik': (['horror', 'thriller'], ['action', 'musical']),
        'Gizemli': (['mystery', 'crime', 'thriller'], []),
        'Geek': (['sci-fi', 'fantasy', 'animation'], []),
        'Dans': (['musical', 'music'], []),
        'Cocuk': (['animation', 'comedy', 'family', 'musical'], ['adult', 'war', 'horror', 'thriller', 'crime']),
        'Entel': (['biography', 'history', 'documentary', 'film-noir', 'short'], [])
    }

    if mood not in mood_to_genres:
        print("Seni Ruhsuz!")
        return

    selected_genres, excluded_genres = mood_to_genres[mood]
    selected_genres = set(selected_genres)
    excluded_genres = set(excluded_genres)

    def genre_match(genres):
        genre_list = set(map(str.lower, map(str.strip, genres.split(','))))
        return bool(selected_genres.intersection(genre_list)) and not bool(excluded_genres.intersection(genre_list))

    filtered_df = df[df['GENRES'].apply(genre_match)]

    if filtered_df.empty:
        print(f"{mood} ruh haline uygun film bulunamadı.")
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

# Example usage
# random_movies = rec_random_movies(filtre_df, 10) # TEST AMAÇLIDIR







# END FUNCTIONS ##########################################################################################


# all_genres = set(genre for sublist in df['GENRES'].dropna().str.split(',') for genre in sublist) # TEST AMAÇLIDIR




df.info()




# Streamlit interface
st.title("K.E.T.S.")

# Get most popular movies per genre
popular_movies_df = rec_top_all_genres(df)

# Select Movie
movie_choice = st.selectbox("Bir film seç:", popular_movies_df['ORIGINAL_TITLE'])

if st.button("Benzer filmler bul", key="find_similar_button"):
    movie_tconst = popular_movies_df[popular_movies_df['ORIGINAL_TITLE'] == movie_choice]['TCONST'].values[0]
    similar_movies = get_similar_by_id(movie_tconst)
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