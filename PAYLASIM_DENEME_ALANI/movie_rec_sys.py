import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# df = pd.read_csv('src/movies/datasets/movies.csv', low_memory=False) # DtypeWarning kapatmak icin
df = pd.read_csv('../src/movies/datasets/movies.csv', low_memory=False)
df['GENRES'] = df['GENRES'].str.lower()
##############################################################################
# TEMIZLEME ISLEMLERI - ARTİK GEREK YOK AMA FİKİR İCİN BİRAKTİM
##############################################################################
#df = df.drop(columns=['Unnamed: 0', 'POPULARITY', 'OVERVIEW', 'POSTER_PATH', 'TIME_MINUTES'])
#df = df[df['IS_ADULT'] != 1]
#df = df.drop(columns=['IS_ADULT'])
#df.replace('\\N', np.nan, inplace=True)
#df.isnull().sum()
## Drop rows with NaN values in all columns
#df = df.dropna()
## Reset index
#df = df.reset_index(drop=True)

#df['YEAR'] = pd.to_datetime(df['YEAR'], format='%Y', errors='coerce')

#df.to_csv('src/movies/datasets/movies.csv', index=False)
##############################################################################

df.shape # (249804, 8)

df.info()
df.head()

#  #   Column          Non-Null Count   Dtype
# ---  ------          --------------   -----
#  0   TCONST          249804 non-null  object          ---- IMDB IDsi
#  1   ORIGINAL_TITLE  249804 non-null  object          ---- Filmin ismi
#  2   TYPE            249804 non-null  object          ---- Medya tipi, dizi, film, belgesel vs.
#  3   AVG_RATING      249804 non-null  float64         ---- IMDB nin kendi hesapladigi average puan
#  4   VOTE_COUNT      249804 non-null  int64           ---- Oy sayisi
#  5   GENRES          249804 non-null  object          ---- Turler, komedi, romantik vs.
#  6   DIRECTORS       249804 non-null  object          ---- Yonetmen isimleri
#  7   YEAR            249804 non-null  datetime64[ns]  ---- Filmin tarihi
# dtypes: float64(1), int64(1), object(6)

# df['TYPE'].value_counts()

# TYPE
# tvEpisode       121696
# movie            87876
# short            11159
# tvMovie          10701
# tvSeries          8329
# tvMiniSeries      3223
# video             2789
# videoGame         1987
# tvSpecial         1783
# tvShort            261
# Name: count, dtype: int64
# tv_series_df = df[(df['TYPE'] == 'tvSeries') & (df['VOTE_COUNT'] > 5000)]
# tv_episodes_df = df[(df['TYPE'] == 'tvEpisode') & (df['ORIGINAL_TITLE'] == 'Tokyo Ghoul: re')]
# Split GENRES if needed
# Gerekirse GENRES i parcalama kodu
# all_genres = set(genre for sublist in df['GENRES'].dropna().str.split(',') for genre in sublist)
# {'action',
#  'adult',
#  'adventure',
#  'animation',
#  'biography',
#  'comedy',
#  'crime',
#  'documentary',
#  'drama',
#  'family',
#  'fantasy',
#  'film-noir',
#  'game-show',
#  'history',
#  'horror',
#  'music',
#  'musical',
#  'mystery',
#  'news',
#  'reality-tv',
#  'romance',
#  'sci-fi',
#  'short',
#  'sport',
#  'talk-show',
#  'thriller',
#  'war',
#  'western'}

##################################################################################################
# FILTRELEME YONTEMI ILE TAVSIYE KODLARI
##################################################################################################
def recommend_top(df, genre='Comedy', media_type='movie', count=1, vote_threshold = 500):
    # Filter the dataset by the selected genre and type
    # GENRE ve TYPE e gore filtrele
    # VOTE_COUNT u belli bir sayidan fazla olsun
    genre_filter = df['GENRES'].str.contains(genre, case=False, na=False)
    type_filter = df['TYPE'].str.contains(media_type, case=False, na=False)
    vote_count_filter = df['VOTE_COUNT'] > vote_threshold
    filtered_df = df[genre_filter & type_filter & vote_count_filter]

    # Sort the filtered dataset by average rating in descending order and select the top 10
    # Filtrelenmis datasetini AVG_RATING e gore sirala ve en bastan 10 tane getir
    top_10 = filtered_df.sort_values(by='AVG_RATING', ascending=False).head(count)

    # Select relevant columns to display
    top_10_recommendations = top_10[['TCONST', 'ORIGINAL_TITLE', 'TYPE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']]

    return top_10_recommendations

# Example usage: Recommend top 10 popular movies for the genre 'Horror', vote count > 50k
# Ornek kullanim: en populer 10 adet Horror turu ve oy sayisi 50binin uzerinde
recommend_top(df, 'comedy', 'movie', count= 10, vote_threshold=50000)


def recommend_most_popular_per_genre(df):
    # Create an empty list to store recommendations
    # Tavsiye icin bos bir liste tanimla
    recommendations = []

    # Get all unique genresS
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
        columns_to_display = ['TCONST', 'ORIGINAL_TITLE', 'TYPE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']
        recommendations_df = recommendations_df[[
            col for col in columns_to_display if col in recommendations_df.columns
        ]]

    return recommendations_df


# Kullanim:
print(recommend_most_popular_per_genre(df))


# Shrin dataframe for cosine sim
#filt_df = df[(df['VOTE_COUNT'] > 3000) & (df['AVG_RATING'] > 6)]
#filt_df = filt_df.reset_index(drop=True)
#filt_df.shape

#Text alanlarını birleştirme
df['combined_features'] = df['ORIGINAL_TITLE'] + ' ' + df['GENRES'] + ' ' + df['DIRECTORS']
# Replace periods (.) with empty strings and commas (,) with spaces
df['combined_features'] = df['combined_features'].str.replace('.', '', regex=False)
df['combined_features'] = df['combined_features'].str.replace(',', ' ', regex=False)
df['combined_features'] = df['combined_features'].str.lower()
df.head()

#########################################################
# Shrink dataframe for cosine sim - BURADA KIRPMAK ZORUNDA KALDİM
filt_df = df[(df['VOTE_COUNT'] > 2000) & (df['TYPE'] == 'movie')]

# reset index cunku out of bounds hatasi veriyor sonra
filt_df = filt_df.reset_index(drop=True)
filt_df.shape



##################################################################################################
# ICERIK TEMELLI FILTRELEME YONTEMI ILE TAVSIYE KODLARI - GENRES
##################################################################################################
# GENRES KISMININ MATEMATIKSEL OLARAK TEMSILI ICIN METIN VEKTORLESTIRME

# Tek basina anlam tasimayan ingilizce kelimeleri cikar orn: and, or, of vs.
#tfidf = TfidfVectorizer(stop_words='english')

# TF-IDF ile özellik çıkarma - Dask kullanarak

# Tek basina anlam tasimayan ingilizce kelimeleri cikar orn: and, or, of vs.
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

# Ayni isimdekileri sil sadece sonuncusunu birak
indices = indices[~indices.index.duplicated(keep='last')]

movie_index = indices['Se7en']

def get_similar_movies(movie_index, cosine_sim, df, top_n=5):
    if 0 <= movie_index < len(cosine_sim):
        similarity_scores = list(enumerate(cosine_sim[movie_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i for i, _ in similarity_scores[1:top_n+1]]  # Exclude the movie itself
        return df.iloc[movie_indices]
    else:
        return f"Index {movie_index} is out of bounds."

# Example usage
movie_index = indices['Se7en'] # Replace with the index of the movie you want to find similarities for
similar_movies_df = get_similar_movies(movie_index, cosine_sim, df)

print(similar_movies_df)





# A CODE THAT CAN FIND TCONST BY MOVIE NAME AND VICE VERSA
# FILTRELERI HER TUR ICIN AYRI YAP SONRA VERILEN PARAMETREYE GORE ILGILI FONKSIYONU CAGIRAN YAZ
# RUH HALI - MOD KOLONLARI OLUSTUR 'BUGUN NASIL HISSEDIYORSUN'
# YEAR YENI BIR KATEGORI ICIN KULLANILABILIR TYPE A GORE VE RATING

# Using GENRES column, create one emotional state name and put it accordingly to EMO_STAT column for each row


# mood_to_genres = {
#    'happy': ['Comedy', 'Family'],
#    'sad': ['Drama', 'Romance'],
#    'excited': ['Action', 'Adventure'],
#   'scared': ['Horror', 'Thriller'],
#    'curious': ['Documentary', 'Biography']
#}

#def get_genres_for_mood(mood):
#    return mood_to_genres.get(mood, [])

#def recommend_movies(mood, df, top_n=10):
#    genres = get_genres_for_mood(mood)
#    if not genres:
#        return []
#
#    filtered_df = df[df['GENRES'].apply(lambda x: any(genre in x for genre in genres))]
#    recommended_movies = filtered_df.sort_values(by=['AVR_RATING', 'VOTE_COUNT'], ascending=False).head(top_n)
#    return recommended_movies

#Example usage:
#mood = 'happy'  # Replace with user's mood
#recommendations = recommend_movies(mood, movies_df)
#print(recommendations)