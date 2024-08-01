
#Importing
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Load the dataset
df = pd.read_csv('/Users/avokado/PycharmProjects/miuul/Final/datasets/movies_31-tem.csv', low_memory=False)



# TODO: GENEL BAKIS
def check_detail(dataframe):
    d = {'SHAPE': dataframe.shape,
         'COLUMNS': dataframe.columns,
         'INDEX': dataframe.index,
         'VALUE TYPES': dataframe.dtypes,
         'DUPLICATED VALUES': dataframe.duplicated().sum(),
         'NUMBER OF UNIQUE VALUES': dataframe.nunique(),
         'ANY MISSING VALUES': dataframe.isnull().values.any(),
         'MISSING VALUES': dataframe.isnull().sum(),
         'DESCRIBE.T': dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T}
    hashtags = '---------------------------'
    for key, val in d.items():
        print(f'{hashtags} {key} {hashtags}')
        print(val)
    print(f'{hashtags} {"LIST END"} {hashtags}')

check_detail(dataframe=df)



# TODO: DUPLICA TEMIZLIGI  : remove_and_show_duplicates :Direk silme
'''
ACIKLAMA:
60 tane duplica satir var.Ya gözden kacti ya da belirli bir amac icin birakildi.
Tegit etmekte fayda var.
Hatta direk böyle silip cvs dosyasi olarak kaydedip en son kodlari o dosya üzerinden okutabiliriz
'''
def remove_and_show_duplicates(dataframe, keep_option='first'):
    """
    Yinelenen satırları tespit edip görüntüleyen ve silen fonksiyon.
    dataframe (pd.DataFrame): Yinelenen satırların tespit edileceği ve silineceği DataFrame.
    keep_option (str): Hangi kopyayı tutmak istediğinizi belirler. 'first' ilk kopyayı, 'last' son kopyayı,
                       'none' tüm kopyaları kaldırır.

    Döndürülen ise : pd.df: Yinelenen satırlar kaldırılmış df.
    """
    if keep_option not in ['first', 'last', 'none']:
        raise ValueError("keep_option 'first', 'last', veya 'none' olmalıdır.")

    # Yinelenen satırları görüntüleme
    duplicate_rows = dataframe[dataframe.duplicated()]
    print("\nYinelenen Satırları Görüntüleme (İlk Kopya Hariç):\n", duplicate_rows)

    if keep_option == 'none':
        # Tüm kopyaları kaldırma
        df_no_duplicates = dataframe.drop_duplicates(keep=False)
        print("\nTüm Yinelenen Satırları Kaldırma:\n", df_no_duplicates)
        return df_no_duplicates
    else:
        # Belirli bir kopyayı tutarak yinelenen satırları kaldırma
        df_unique = dataframe.drop_duplicates(keep=keep_option)
        print(f"\nYinelenen Satırları Kaldırma ({keep_option.capitalize()} Kopya Hariç):\n", df_unique)
        return df_unique

# Duplice Satırları Kaldırma ve Gösterme (ilk kopya hariç)
df = remove_and_show_duplicates(df, keep_option='first')





#TODO: Fonksiyon 1 : recommend_top :Belirli bir genre ve belirli bir type icin en yüksek oy alan filmleri öneren bir filtreleme ve siralama yapiyor.

'''
ACIKLAMA 1:
Artik net data setimizde tek bir type var oda movie, bu noktada media_type parametre özelligi 
burda herhangi bir fayda saglamiyor yani anlam ifade etmiyor. Gereksiz bir filtreleme oluyor.
Eğer veri setimizde sadece filmler varsa ve TYPE sütunu artık bir anlam ifade etmiyorsa, 
media_type parametresine gerek yoktur. Bu durumda, kodu basitleştirmek ve gereksiz kontrolleri kaldırmak en iyisi olacaktır. 
Cünkü episodi ya da short ,tv_series gibi typeler yok artik, bu nedenle media_type parametresini cikartabiliriz.

Ama eger ekstra dizi önermek gibi bir gelisime zaman kalirsa ya da istersek, durabilirde. düsüncem, sadelestirmekte her zaman fayda var .

Bu durumda eger yukarida yazilanlari yaparsak , sadece genresler arasinda bir icerigin  en yüksek oy alana göre tasarlanmis 
ve sadece genresler arasinda olucak. typesiz.Alternatif olarak  revize bir  fonksiyon birakacagim.

Özellikler: sadece en yüksel oy sayisini baz aliyor.
'''
def recommend_top(df, genre='Comedy', media_type='movie', count=1, vote_threshold=500):
    # Filtreleme işlemleri
    genre_filter = df['GENRES'].str.contains(genre, case=False, na=False)
    type_filter = df['TYPE'].str.contains(media_type, case=False, na=False)
    vote_count_filter = df['VOTE_COUNT'] > vote_threshold

    # Filtrelerin uygulanması
    filtered_df = df[genre_filter & type_filter & vote_count_filter]

    # Sıralama ve en iyi sonuçların alınması
    top_10 = filtered_df.sort_values(by='AVG_RATING', ascending=False).head(count)

    # Geri döndürülecek sütunlar
    top_10_recommendations = top_10[['TCONST', 'ORIGINAL_TITLE', 'TYPE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']]

    return top_10_recommendations

recommend_top(df, 'comedy', 'movie', count= 10, vote_threshold=50000) # degistirilebilir özellikler.

# TODO: Alternatif Revize Kod 1
def recommend_top_alt(df, genre='Comedy', count=1, vote_threshold=500):
    # Filtreleme işlemleri
    genre_filter = df['GENRES'].str.contains(genre, case=False, na=False)
    vote_count_filter = df['VOTE_COUNT'] > vote_threshold

    # Filtrelerin uygulanması
    filtered_df = df[genre_filter & vote_count_filter]

    # Sıralama ve en iyi sonuçların alınması
    top_10 = filtered_df.sort_values(by='AVG_RATING', ascending=False).head(count)

    # Geri döndürülecek sütunlar
    top_10_recommendations = top_10[['TCONST', 'ORIGINAL_TITLE', 'TYPE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']]

    return top_10_recommendations

recommend_top_alt(df, genre='Comedy', count=1, vote_threshold=500)
'''
Alternatif Kod Ciktisi 1:
20566  tt0252487  Hababam Sinifi  movie       9.200       43271  Comedy,Drama
'''


top_recommendations = recommend_top_alt(df, genre='Comedy', count=2, vote_threshold=500) #Deneme amacli cikartilacaktir.
print(top_recommendations)
'''
Deneme Kod Ciktisi:
          TCONST       ORIGINAL_TITLE   TYPE  AVG_RATING  VOTE_COUNT        GENRES
20566  tt0252487       Hababam Sinifi  movie       9.200       43271  Comedy,Drama
20789  tt0267277  Ashi Hi Banwa Banwi  movie       9.000        2003        Comedy
'''




#TODO: Fonksiyon 2 : recommend_most_popular_per_genre  :Her genre icin en popüler (en yüksek oy sayusu ve ortalama puana sahip) icerigi öneriyor.
'''
ACIKLAMA 2:
Bu fonksiyonda her bir genre icin tek bir tane önerecegi anlami cikiyor. 
Fakat dönen ciktida her bir genre icin en yüksek oy sayisi ve en yüksek ortalama esas alinarak 24 ayri tavsiye filmleri döndüruyor.
Toplamda 24 secenek streamlit te tavsiye olarak sunulmasi ekranda asiri kalabaliklasmaya ve arka planda gereksiz islem sayisi gibi bisi olusuyor.
Revizeye gidilmeli diye düsünüyorum.Tek bir öneri olarak sunulabilir yada ilk bes seklinde. 
Yada 2 li karsilastirmayla siralama yapabiliriz. Yada hic ugrasmamak adina direk yukaridaki gibi ilk bes ya da 3 gidi olabilir.
Ve ek olarak genreler bazi satirlarda tek bazilarinda 2-3 kombin seklinde. Dolayisiyla ciktida fazla tekrarlayan genre gürültüsüde olusuyor. 
Yani mantiken aslinda her bir genre icin öneride bulunmuyorr gibi bir sey görünüyor bu konbin durumundan dolayi.
Alternatif kodda en kolay revize yolu olarak bir kod biraktim. ama farkli bir yolda izlenilebilir. kombinlerle alakli.

Özelliker: En yüksek oy sayisi ve ortalamayi esas aliyor.
'''
def recommend_most_popular_per_genre(df):
    # Tavsiye icin bos bir liste tanimla
    recommendations = []

    # Tum virgul ile ayrilmis genreleri tek tek al
    all_genres = set(genre for sublist in df['GENRES'].dropna().str.split(',') for genre in sublist)

    for genre in all_genres:
        # genre basina veri setini filtrele
        genre_filter = df['GENRES'].str.contains(genre, case=False, na=False)
        filtered_df = df[genre_filter]

        if not filtered_df.empty:
            # Genre icin en vote_countu ve avg_rating i yuksek olanlari diz ve birinci elemani al
            most_popular = filtered_df.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=[False, False]).iloc[0]
            recommendations.append(most_popular)

    # DataFrame'e cevir
    recommendations_df = pd.DataFrame(recommendations)

    # Gereken kolonlarin olup olmadigini kontrol et
    if not recommendations_df.empty:
        # Gosterilecek kolonlari sec
        columns_to_display = ['TCONST', 'ORIGINAL_TITLE', 'TYPE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']
        recommendations_df = recommendations_df[columns_to_display]

    return recommendations_df

print(recommend_most_popular_per_genre(df))

# TODO: Alternatif Revize Kod 2
def recommend_most_popular(df):
    # Veri setini VOTE_COUNT ve AVG_RATING'e göre sıralayarak en popüler içeriği bul
    most_popular = df.sort_values(by=['VOTE_COUNT', 'AVG_RATING'], ascending=[False, False]).head(1)

    # Gereken kolonları seç
    columns_to_display = ['TCONST', 'ORIGINAL_TITLE', 'TYPE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']
    most_popular = most_popular[columns_to_display]

    return most_popular

print(recommend_most_popular(df))
'''
Alternatif Kod Ciktisi 2: head(1) seklinde tek getiriyor. istersek beslide cikartiriz. örnek kod ciktisi buna göre koydum. 

          TCONST            ORIGINAL_TITLE   TYPE  AVG_RATING  VOTE_COUNT                   GENRES
12856  tt0111161  The Shawshank Redemption  movie       9.300     2919274                    Drama
18927  tt0468569           The Dark Knight  movie       9.000     2900164       Action,Crime,Drama
18931  tt1375666                 Inception  movie       8.800     2576105  Action,Adventure,Sci-Fi
15567  tt0137523                Fight Club  movie       8.800     2351755                    Drama
9870   tt0109830              Forrest Gump  movie       8.800     2283036            Drama,Romance
'''






# TODO: Fonksiyon 3 : get_similar_movies :Icerik temelli Öneri sistemi. (Metinsel benzerlikleri esas aliyor.)

# Ön Hazirlik
content_df = df.copy() # Isim degisikligi yapilmistir.

# TODO: Performans Cagrisi ! 86 saniye sürdü , hesapladim.:D
'''
DOKTOR BUNA BIR CARE :D
filt_df = content_df (isim degisikligi)
# Shrink dataframe for cosine sim - BURADA KIRPMAK ZORUNDA KALDİM
filt_df = df[(df['VOTE_COUNT'] > 2000) & (df['TYPE'] == 'movie')]

# reset index cunku out of bounds hatasi veriyor sonra
filt_df = filt_df.reset_index(drop=True)
filt_df.shape


30 Temmuz aksamindaki kod dosyasinda yukaridaki kod vardi- suanki app.py  yani 31 temmuzlu en son kod dosyasinda yok, 
iki farkli dosyadada bu adimi denedim. Yukaridaki kodlar olunca daha hizli calisti benim pc de.
Bu gözlenmeli - eklenmeli- yada bunu konusalim. 
'''

tfidf = TfidfVectorizer(stop_words='english')
# TF-IDF Matrisinin olusturulmasi
tfidf_matrix = tfidf.fit_transform(content_df['COMBINED_FEATURES'])
# Cosine Similarity Matrisinin Olusturulmasi
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Benzerliklere gore onerilerin yapilmasi
indices = pd.Series(content_df.index, index=content_df['ORIGINAL_TITLE'])



'''
ACIKLAMA:
Biraz revize ve detay ekledim fonksiyonun anlasilmasi acisindan. Zaten 31 temmuzlu dosya calisiyordu.
Fakar direk TCONST üzerinden gitme kismini burada bir sekilde baglanmasi lazim ? bu bilgiyi nasil alicaz? benzersiz degisken.
Her filmin benzersiz id lerini elle tek tek atayamayiz.atayamazsak ve bu bir özellikse, bu bilgisi müsteriden nasil alicazki öneri yapabilelim?
suan cümleyi toparlayamadim ama konusalim derim.
'''
def get_similar_movies(tconst, cosine_sim, df, top_n=5):
    """
    Belirli bir film tanımlayıcısına (TCONST) dayalı olarak benzer filmleri bulan fonksiyon.

    Parametreler:
    tconst : Benzer filmlerini bulmak istediğiniz filmin TCONST değeri.
    cosine_sim : Filmler arasındaki kosinüs benzerlik matrisini içeren bir matris.
    df : Filmler hakkında bilgileri içeren DataFrame.
    top_n : Döndürülecek en benzer film sayısı. Varsayılan değer 5'tir.

    Döndürülen:
    pd.DataFrame: En benzer `top_n` filmi içeren DataFrame.

    Hata Durumları:
    ValueError: Eğer belirtilen TCONST değerine sahip bir film bulunamazsa veya indeks integer değilse.
    """
    try:
        # Belirtilen TCONST değerine sahip filmin indeksini bulma
        movie_index = df.index[df['TCONST'] == tconst].tolist()[0]
    except IndexError:
        raise ValueError(f"Belirtilen TCONST değerine sahip film bulunamadı: {tconst}")

    if not isinstance(movie_index, int):
        raise ValueError("Movie index is not an integer.")

    # Filmler arasındaki benzerlik skorlarını hesaplama
    similarity_scores = list(enumerate(cosine_sim[movie_index]))

    # Benzerlik skorlarını azalan sırada sıralama
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # En benzer `top_n` filmin indekslerini alma
    movie_indices = [i for i, _ in similarity_scores[1:top_n+1]]

    # Benzer filmleri içeren DataFrame'i döndürme
    return df.iloc[movie_indices]
'''
ACIKLAMA 3:
Biraz revize ve detay ekledim fonksiyonun anlasilmasi acisindan. Zaten 31 temmuzlu dosya calisiyordu.
Fakar direk TCONST üzerinden gitme kismini burada bir sekilde baglanmasi lazim ? bu bilgiyi nasil alicaz? benzersiz degisken.
Her filmin benzersiz id lerini elle tek tek atayamayiz.atayamazsak ve bu bir özellikse, bu bilgisi müsteriden nasil alicazki öneri yapabilelim?
suan cümleyi toparlayamadim ama konusalim derim.
'''

#Benzer filmleri almak için:
similar_movies = get_similar_movies('tt00123', cosine_sim, content_df, top_n=5) # Hata  uyarisi özelligi okey
print(similar_movies)

# Ufak not sizd´ce burda fonksiyonlari cagirirken dbir degiskene atamamiz ya da print yaptirmamiz kesin gerekli mi ? stream lit acisindan falan ?
#Ona göre ya birinde cikarcaz bu kisimlari ya da digerine eklicez gerekliyse.
get_similar_movies('tt0022879', cosine_sim, content_df, top_n=5)  # 'tt0022879' örnek movie id
'''
Deneme Ciktisi : Mesele burdaki Western neden geliyor.digerlerinin benzerligi var gibi .merak ettim bu noktayi.
         TCONST              ORIGINAL_TITLE   TYPE  AVG_RATING  VOTE_COUNT                  GENRES          DIRECTORS        YEAR                                           OVERVIEW                                  COMBINED_FEATURES
1673  tt0025920  Twenty Million Sweethearts  movie       6.300         476  Comedy,Musical,Mystery        Ray Enright  1934-01-01  Unscrupulous agent Pat O'Brien makes singing w...  twenty million sweethearts comedy,musical,myst...
5050  tt0049973  You Can't Run Away from It  movie       5.900         606  Comedy,Musical,Romance        Dick Powell  1956-01-01  A reporter stumbles on a runaway heiress whose...  you can't run away from it comedy,musical,roma...
2644  tt0047550            Susan Slept Here  movie       6.400        2176    Comedy,Drama,Romance      Frank Tashlin  1954-01-01  Suffering from a case of writer's block, scree...  susan slept here comedy,drama,romance frank ta...
1990  tt0040835                Station West  movie       6.600        1530                 Western    Sidney Lanfield  1948-01-01  Dick Powell is a stranger in town battling Ray...  station west western sidney lanfield dick powe...
341   tt0042779           Nancy Goes to Rio  movie       6.400         628  Comedy,Musical,Romance  Robert Z. Leonard  1950-01-01  Mother and daughter (Sothern and Powell) compe...  nancy goes to rio comedy,musical,romance rober...

'''


# TODO: Fonksiyon 4 :  follow_your_mood: Bugün nasil hissediyorsun ?  ya da Havan Nasil ? Cümle degisebilir ama ruh haliyle alakali bir cümle ve fonksiyon.
'''
ACIKLAMA 4: 
Bu kod fikirlerimizden biri.
Ruh hali seceneklerini bol tuttum ve tamamiyle isimlendirmeler sallamasyon , buralara birlikte karar verelim. 
Kac tane olucak ruh hali kategori isimleri ne olucak gibi gibi.
Fakat ruh hali kategori atamalarida önemli biraz netten arastirarak duygu hallerini cogaltarak atamalar yapmaya calistim. 
Duyygu-ruh hallerini az öz ama yaratici yapip kategori atamalarinada yar verebiliriz. 
Fakat burda yine genres lerin bazilar tek bazilari kombin durumu var ama bence sorun olmaz. atamalari mantikli yaparsak.

user_mood = input("Hislerini takip et: ").strip() : asagidaki kodda bu sekilde in putla 
ruh halini sordurup concoleden ruh hali seceneklerinden birini yaziyoruz , sonrada fonksiyonu cagirinca öneriler geliyor.

Özellikler: ruh hali secenegi ya da mood durumuna göre kategoriler olusturup, atamalar yaparak en yüksek oy olan ve ortalamaya siralayip ilk 10 u veriyor.
Bu noktalarada revize yapabiliriz, öneri sayisindaki degisiklik gibi. 
Fakat kullaniciya modu sectirme yada yazdir gibi baglantiyi stream litte nasil yapicaz fikyom hic yok.Ama  ek derste anlatilmisti sanirim.
Baska yolu varsa onuda deneyebiliriz.

'''
def follow_your_mood(df, user_mood):
    mood_to_genres = {
        'Hahaha': ['comedy', 'family', 'animation'],
        'Biraz duygusal': ['drama', 'romance'],
        'Heyecan dorukta': ['action', 'adventure', 'sci-fi', 'fantasy'],
        'Gerilsekte korkmayiz': ['horror', 'thriller'],
        'Hep merakli': ['documentary', 'biography', 'history', 'news', 'sport', 'reality-tv', 'game-show', 'talk-show'],
        'Biraz dans': ['musical'],
        'Entel kus': ['film-noir', 'music', 'short'],
        'Dedektiflik benim isim': ['crime', 'mystery'],
        'Büyüdük artik': ['adult', 'war', 'western']
    }

    if user_mood not in mood_to_genres:
        print("Seni Ruhsuz !")
        return

    selected_genres = mood_to_genres[user_mood]

    def genre_match(genres):
        return any(genre.lower() in genres.lower() for genre in selected_genres)

    filtered_df = df[df['GENRES'].apply(genre_match)]

    if filtered_df.empty:
        print(f"{user_mood} ruh haline uygun film bulunamadı.")
    else:
        filtered_df = filtered_df.sort_values(by=['AVG_RATING', 'VOTE_COUNT'], ascending=[False, False])
        top_10_movies = filtered_df.head(10)

        print("Önerilen Filmler:")
        print(top_10_movies[['TCONST', 'ORIGINAL_TITLE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']])

# Kullanıcının ruh haline göre öneri yapılması Adimlari
print("Mevcut Ruh Halleri: Hahaha, Biraz duygusal, Heyecan dorukta, Gerilsekte korkmayiz, Hep merakli, Biraz dans, Entel kus, Dedektiflik benim isim, Büyüdük artik")
user_mood = input("Hislerini takip et: ").strip()

follow_your_mood(df, user_mood)
''' 
Deneme Ciktisi:
Kategorilerede yaptigim atama sonuclari geliyor. buna overwievde
Hislerini takip et: >? Gerilsekte korkmayiz #####(bu kismi consoleden yaziinca ve fonksiyonu cagirinca calisyor, kastettigim yer burasi )
follow_your_mood(df, user_mood)
Önerilen Filmler:
          TCONST            ORIGINAL_TITLE  AVG_RATING  VOTE_COUNT                   GENRES
20019  tt0214915          Manichitrathazhu       8.700       12603    Comedy,Horror,Mystery
10254  tt0102926  The Silence of the Lambs       8.600     1564755     Crime,Drama,Thriller
6417   tt0407887              The Departed       8.500     1435208     Crime,Drama,Thriller
10888  tt0078748                     Alien       8.500      961342            Horror,Sci-Fi
999    tt0054215                    Psycho       8.500      724950  Horror,Mystery,Thriller
992    tt0047396               Rear Window       8.500      526938         Mystery,Thriller
7562   tt0058625              Suna no onna       8.500       23044           Drama,Thriller
3366   tt0054407                   Le trou       8.500       20675     Crime,Drama,Thriller
18930  tt1345836     The Dark Knight Rises       8.400     1844288    Action,Drama,Thriller
18924  tt0209144                   Memento       8.400     1334034         Mystery,Thriller

'''




#TODO:Fonksiyon 5 :
''' 
ACIKLAMA:
Bu fonksiyonu ilk basta sey diye düsünmüstük typelar  arasinda rastgele  secim. Ama suan tek type mowie o yüzden bu kisma baska bir baglanti yapabiliriz. 
Sadece Overwiev mesela, ya da yönetmen mesela belirli bir  yönetmen sordurup oluru varsa ,  o yönetmenin filmlerinden rastgele olabilir. 
Yada direk  genreslerden . Ayri bir öneri bicimi gibi  daha özele indirgenmis gibi bisi sanirim.
Ama genresle ilgili cok sey yaptik diger fonksiyonlarda belki burda bir farklilik yapabiliriz.
O kismi degisitiroyrum simdilik karar verince  düzeltiriz.  type yerine baska bir sey ekleyebiliriz. 
'''
def recommend_movies_by_type(df):
    # Veriyi TYPE sütununa göre grupla
    grouped_df = df.groupby('GENRES')

    recommendations = []

    # Her grup için rastgele 10 öneri al
    for type_name, group in grouped_df:
        # Rastgele 10 film seç
        random_10_movies = group.sample(n=min(2, len(group)))

        # Seçilen filmleri listeye ekle
        recommendations.append({
            'GENRE': type_name,
            'Movies': random_10_movies
        })

    # Sonuçları yazdır
    for rec in recommendations:
        print(f"Tür: {rec['GENRE']}")
        if not rec['Movies'].empty:
            print(rec['Movies'])
        else:
            print("Bu türde yeterli film bulunamadı.")
        print()
recommend_movies_by_type(df)



# TODO: Fonksiyon 6: recommend_top_directors_movies: En cok izlene yönetmene göre rastgele öneri.

'''
ACIKLAMA 6:

Bu kodda her bir yönetmenden rastgele 10 film önerisinde bulunarak en çok izlenen beş yönetmene göre film önerisi yapar. 
Bu yöntemler kullanicilarin - izleyicilerin favori yönetmenlerinin popüler filmlerini keşfetmeleri sağlanır.
Rastgele secim yerine yine bir belirli duruma göre siralamada olur.'''

def recommend_top_directors_movies(df):
    """
    En çok izlenen beş yönetmene göre rastgele 10 film önerisi yapan fonksiyon.

    Parametreler:
    df (pd.DataFrame): Filmler hakkında bilgileri içeren DataFrame.

    Döndürülen:
    None
    """
    # Yönetmen başına toplam izlenme (oy) sayısını hesapla
    director_vote_counts = df.groupby('DIRECTORS')['VOTE_COUNT'].sum()

    # En çok izlenen beş yönetmeni seç
    top_5_directors = director_vote_counts.nlargest(5).index

    recommendations = []

    for director in top_5_directors:
        # Yönetmen bazında filtreleme
        director_df = df[df['DIRECTORS'] == director]

        # Rastgele 10 film seç
        random_10_movies = director_df.sample(n=min(3, len(director_df)))

        # Seçilen filmleri listeye ekle
        recommendations.append({
            'DIRECTOR': director,
            'Movies': random_10_movies
        })

    # Sonuçları yazdır
    for rec in recommendations:
        print(f"Yönetmen: {rec['DIRECTOR']}")
        if not rec['Movies'].empty:
            print(rec['Movies'][['TCONST', 'ORIGINAL_TITLE', 'AVG_RATING', 'VOTE_COUNT', 'GENRES']])
        else:
            print("Bu yönetmene ait yeterli film bulunamadı.")
        print()
# En çok izlenen beş 5 yönetmene göre  rastgele 3 film seklinde. Bunuda degistirebiliriz.
recommend_top_directors_movies(df)





# TODO: Fonksiyon 7: degisken ismi loading ... :D  Zaman göre Nostalji retro yada yillar gibi seceneklere göre öneri.
'''
ACIKLAMA 7:
Burada YEAR  degiskenini  kullanmaya calisiyorum fakat biraz karisik, üzerinde calisiyorum.Fikir ve destekle bisiler cikabilir.
Ama yok bunlar yeter diyorsanizda salariz (:
'''

# TODO: GENEL ACIKLAMA: HER ÖNERI ISE YARIYOR AMA GENEL MANADA BIR OYNAMA YAPABILIRSEK SÜPER OLUR. NOKTADA BU GENRES KOLONLARININ KOMBIN OLUSU.

# VEEEEE TASLAAK OLUSTU. SÜKÜRLER OLSUN. HEPIMIZE TESEKKÜRLER (: KIRIMIZI BALON.

# Referans Kod
def recommend_movies_by_nostalgic_category(df):
    # Nostaljik kategorilere göre tarih aralıkları tanımla
    nostalgic_categories = {
        'Nostalji': (1960, 1979),
        'Retro': (1980, 1989),
        'Mazi': (1990, 1999),
        'Yeniler': (2000, 2009),
        'Günümüz': (2010, 2020)
    }

    # Kullanıcıya kategorileri sun
    print("Mevcut Nostaljik Kategoriler:")
    for category in nostalgic_categories:
        print(category)

    # Kullanıcıdan kategori seçmesini iste
    user_category = input("Bir kategori seçin: ").strip()

    # Seçilen kategorinin geçerli olup olmadığını kontrol et
    if user_category not in nostalgic_categories:
        print("Geçersiz kategori seçimi!")
        return

    # Seçilen kategoriye göre başlangıç ve bitiş yıllarını al
    start_year, end_year = nostalgic_categories[user_category]

    # YEAR sütununu integer veri tipine dönüştür
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')

    # Veriyi filtrele
    filtered_df = df[(df['YEAR'] >= start_year) & (df['YEAR'] <= end_year)]

    if filtered_df.empty:
        print(f"{user_category} kategorisinde film bulunamadı.")
    else:
        # Kolon adlarını kontrol et ve sıralama
        if 'AVG_RATING' not in filtered_df.columns or 'VOTE_COUNT' not in filtered_df.columns:
            print("Gerekli sütunlar mevcut değil!")
            return

        filtered_df = filtered_df.sort_values(by=['AVG_RATING', 'VOTE_COUNT'], ascending=[False, False])
        top_10_movies = filtered_df.head(10)

        print(f"{user_category} kategorisindeki önerilen filmler:")
        print(top_10_movies)

recommend_movies_by_nostalgic_category(df)

