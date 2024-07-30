
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Film ve müzik veri setlerini oku
films = pd.read_csv('src/movies/datasets/movies.csv')
games = pd.read_csv('src/games/datasets/games.csv')
games = games[games['Genres'].notna() & (games['Genres'] != '')]
films.head()
games.head()
games.info()



# TF-IDF vektörizer oluştur ## Bu Vektörize ediyor.
tfidf_vectorizer = TfidfVectorizer()

# Film ve müzik türlerini birleştir
film_genre_matrix = tfidf_vectorizer.fit_transform(films['GENRES'])
games_genre_matrix = tfidf_vectorizer.transform(games['Genres'])

# Benzerlik hesaplama
cosine_similarities = linear_kernel(games_genre_matrix, film_genre_matrix)

#Film icin Müzik önerisi
def recommend_game_for_film(film_title, top_n=5):
    # Film başlığına göre film indeksini bul
    film_idx = films[films['ORIGINAL_TITLE'] == film_title].index[0]

    # Belirli film için benzerlik skorlarını al
    sim_scores = list(enumerate(cosine_similarities[:, film_idx]))

    # Benzerlik skorlarına göre sıralama
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # En iyi n müziği seç
    sim_scores = sim_scores[:top_n]
    games_indices = [i[0] for i in sim_scores]

    # Önerilen müzikler
    return games.iloc[games_indices]

# Örnek öneri
recommend_game_for_film('Godfather')

#Müzik icin Film önerisi

def recommend_film_for_music(track_name, top_n=5):
    # Şarkı adına göre şarkı indeksini bul
    music_idx = music[music['track_name'] == track_name].index[0]

    # Belirli şarkı için benzerlik skorlarını al
    sim_scores = list(enumerate(cosine_similarities[music_idx, :]))

    # Benzerlik skorlarına göre sıralama
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # En iyi n filmi seç
    sim_scores = sim_scores[:top_n]
    film_indices = [i[0] for i in sim_scores]

    # Önerilen filmler
    return films.iloc[film_indices]

# Örnek öneri
recommend_film_for_music('Horizon')

#########################################################################################################
#digerinden kirpilmis kod


def create_user_movie_df():
    import pandas as pd
    df = pd.read_csv('src/movies/datasets/movies.csv')
    rating = df['AVERAGERATING']
    df_movies = df[df['TITLETYPE'].isin(['movie', 'tvMovie'])]
    num_votes = df_movies[['TCONST','ORIGINALTITLE','NUMVOTES']]
    rare_movies = num_votes[num_votes["NUMVOTES"] <= 1000]['TCONST']
    common_movies = df_movies[~df_movies["TCONST"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["TCONST"], columns=["ORIGINALTITLE", "GENRES"], values="AVERAGERATING")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Titanic", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)
