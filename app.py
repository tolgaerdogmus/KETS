import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset
df = pd.read_csv('src/movies/datasets/movies_31-tem.csv', low_memory=False)

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

#TODO: SONRA DEGISTIRILEBILIR DF ISMI FONKSIYONLARDA KULLANILDIGI GIBI BIRAKIYORUM SON TEMIZLIKTE DUZENLENIR

filt_df = df.copy()

# Shrink dataframe for cosine sim
#filt_df = df[(df['VOTE_COUNT'] > 2000) & (df['TYPE'] == 'movie')]
#filt_df = filt_df.reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words='english')


# TF-IDF Matrisinin olusturulmasi
tfidf_matrix = tfidf.fit_transform(filt_df['COMBINED_FEATURES'])


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