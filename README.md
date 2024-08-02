# K.E.T.S.

## Veri setlerimiz:
[Veri setleri klasörü](https://drive.google.com/drive/folders/10t8OYe4i0U_OOEMwjhQNM7B3gT5KqObn)




# K.E.T.S. - BETA versiyon
CHANGE LOGS
(öncesi -> sonrası)

### Amaçlar:
- Daha kısa, net ve anlaşılır değişken ve fonksiyon isimlendirmeleri,
- Optimizasyon; kodların ve verisetinin performans artırımı
- Yeni fonksiyonların eklenmesi
## Fonksiyon Değişiklikleri:
-  **recommend_top** değişiklikleri:
	- İsim değişikliği - recommend_top -> rec_top_by_genre
	- Silindi - movie_type = movie | ve ilgili kodları	 
	- Silindi - getirdiği kolonlardan TYPE kaldırıdı
	- İsim değişikliği - top_10 -> top_movies
	- İsim değişikliği - top_10_recommendations -> top_recommendations_by_genre
-  **recommend_most_popular_per_genre** değişiklikleri:
	- İsim değişikliği - recommend_most_popular_per_genre -> rec_top_all_genres
	- Silindi - getirdiği kolonlardan TYPE kaldırıdı
- **recommend_most_popular** değişiklikleri: 
	- İsim değişikliği - recommend_most_popular -> rec_most_popular
	- Eklendi - count parametresi eklenip varsayılan değeri 1 yapıldı. Yeni kullanım şekli rec_most_popular(df) ya da rec_most_popular(df, count=2) istenilen miktar yazılabilir.
- **get_similar_movies** değişiklikleri:
	- İsim değişikliği - get_similar_movies -> get_similar_by_id
	- Eklendi - get_similar_by_title bu da TCONST yerine TITLE benzerliğini alıp ona göre arıyor örneğin: "itani" yazılsa bile bunun en yakını olarak Titanic i eşleştirip, Titaniğe benzer filmler buluyor. 
	- İsim değişikliği: top_n -> count
- **follow_your_mood** değişiklikleri:
	- Optimizasyon: genre_match fonksiyonu içinde set, map kullanıldı performans için ```genre_list = set(map(str.lower, map(str.strip, genres.split(','))))```
	- Özellik eklendi: Fonksiyon artık bir moda göre getirirken, hariç tutacağı kategorilere de bakıyor. Örneğin: Karanlık türünde horror ve thriller getirirken, Batman ve Kill Bill filmleri de getiriyordu çünkü onlar da thriller fakat bir The Sixth Sense gibi bir thriller değil. Şu an çok daha sağlıklı sonuçlar veriyor. Bu sayede gelen sonuçlar çok daha sağlıklı hale getirilebilir. Bu kategoriler için de konuşulması gerekir.
	- Eskiden son filtrede ilk AVG_RATING ve arkasından VOTE_COUNT a göre filtreliyordu. Bunların yerleri değiştirildi. Çünkü AVG_RATING i ilk baz alıyordu ve bunda da az oy kullanılmış filmler araya kaçıyordu. VOTE_COUNT ilk daha çok ilgi görmüş filmleri süzüp sonra da AVG_RATING kullanarak onların da içinden puanı yüksekleri diziyor. ```movies_for_mood = filtered_df.nlargest(count, ['VOTE_COUNT', 'AVG_RATING'])```
	- Eklendi: sona return eklendi programatik olarak başka fonksiyonların (streamlit) da kullanabileceği  hale gelmesi için, ekrana print alma silindi.
- **recommend_movies_by_type**
	- Yanlışlıkla TYPE denilmiş fakat GENRE ye göre çalışıyor, isim değişikliği gerekti
	- Amacı anlaşılamadı, dışarıdan tür seçmiyor fakat içeride rastgele tür e göre rastgele filmler getiriyor aslında özet olarak sadece rastgele x sayıda bir film getirmiş oluyor. Belki dışarıdan tür kabul eden cinsi düşünülebilir sonrasında
- **rec_random_movies** - eklendi
	- filtrelenmiş verisetinden rastgele istenilen adette film listeliyor ve liste sonucu gelen filmleri de kendi içinde oy sayısı ve avg rating ile ekrana listeliyor
	

## Genel Kod Değişiklikleri:
- İsim değişikliği - filt_df -> filtre_df
- Eklendi - TF-IDF e sokulacağı için performans artırmak amacıyla ``df[(df['VOTE_COUNT'] > 2000) & (df['AVG_RATING'] > 6.0)]`` şeklinde filtrelenerek "filtre_df" şeklinde yeni bir ikinci dataframe oluşturuldu. Performans sonuçlarına göre bu filtre ayarlanabilir
- Değişiklik - follow_your_mood fonksiyonu içindeki mood_to_genres isimleri tek kelimeye düşürüldü çünkü kodda çağırırken Türkçe karakter ve boşluklar olmamalı, bu tarz isimlendirmeyi kod içinde değil, dışarıda kullanıcının göreceği yerlerde yapılmalı. (Sayfada şunu seç bunu seç gibi olan başlık ve etiketlerde)
- Değişiklik - mood_to_genres ruh hali isimleri ve yeniden düzenlenmiş kategoriler: (son karar değildir değiştirilebilir)
	- Huzurlu : comedy, adventure, family, animation
	- Duygusal: drama, romance, family
	- Hareketli: action, war, thriller, crime, adventure, western, sport
	- Karanlik: horror, thriller
	- Gizemli: mystery, crime, thriller
	- Geek: sci-fi, fantasy, animation
	- Dans: musical, music
	- Cocuk: animation, comedy, family, musical
	- Entel: biography, history, documentary, film-noir, short
	- ..
	- Veri setindeki şu şekildedir: Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, News, Reality-TV, Romance, Sci-fi, Sport, Thriller, War, Western.
	


## Genel UX Değişiklikleri
(çalışma devam ediyor)