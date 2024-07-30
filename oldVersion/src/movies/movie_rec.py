import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv('src/movies/datasets/movies.csv')
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

df.info()

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

df['TYPE'].value_counts()

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

# Split GENRES if needed
# Gerekirse GENRES i parcalama kodu
# all_genres = set(genre for sublist in df['GENRES'].dropna().str.split(',') for genre in sublist)
# {'Action',
#  'Adult',
#  'Adventure',
#  'Animation',
#  'Biography',
#  'Comedy',
#  'Crime',
#  'Documentary',
#  'Drama',
#  'Family',
#  'Fantasy',
#  'Film-Noir',
#  'Game-Show',
#  'History',
#  'Horror',
#  'Music',
#  'Musical',
#  'Mystery',
#  'News',
#  'Reality-TV',
#  'Romance',
#  'Sci-Fi',
#  'Sport',
#  'Talk-Show',
#  'Thriller',
#  'War',
#  'Western'}

##################################################################################################
# TAVSIYE KODLARI
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
recommend_top(df, 'Horror', 'movie', count= 10, vote_threshold=50000)



def recommend_most_popular_per_genre(df):
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
        columns_to_display = ['TCONST', 'ORIGINAL_TITLE', 'TYPE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']
        recommendations_df = recommendations_df[[
            col for col in columns_to_display if col in recommendations_df.columns
        ]]

    return recommendations_df


most_popular_per_genre = recommend_most_popular_per_genre(df)
print(most_popular_per_genre)
