# Game Data Set Description

This dataset contains information about various video games. Each entry includes the following fields:

### AppID
- **Type**: `string`
- **Description**: Unique identifier for each app | Benzersiz oyun ID.

### Name
- **Type**: `string`
- **Description**: Game name | Oyun ismi.

### Release date
- **Type**: `string`
- **Description**: Release date of the game | Çıkış tarihi.

### Estimated owners
- **Type**: `string` (e.g.: "0 - 20000")
- **Description**: Approximate number of owners of the game | Yaklaşık oyuna sahip kişi sayısı.

### Peak CCU
- **Type**: `int`
- **Description**: Number of concurrent users, yesterday | Bir gün önceki aynı anda oyun oynayan kişi sayısı.

### Required age
- **Type**: `int`
- **Description**: Age required to play, 0 if it is for all audiences | Oynamak için gereken yaş sınırı, 0 ise genele uygun.

### Price
- **Type**: `float`
- **Description**: Price in USD, 0.0 if it's free | Fiyat USD, 0.0 ise ücretsiz.

### DLC count
- **Type**: `int`
- **Description**: Number of DLCs, 0 if you have none | Ek içerik sayısı, 0 ise yok.

### About the game
- **Type**: `string`
- **Description**: Detailed description of the game | Detaylı açıklama.

### Supported languages
- **Type**: `string`, `list`
- **Description**: Supported languages | Desteklenen yazı dilleri.

### Full audio languages
- **Type**: `string`, `list`
- **Description**: Languages with audio support | Desteklenen ses dilleri.

### Reviews
- **Type**: `string`
- **Description**: Reviews of the game | Yapılan eleştiriler, yorumlar.

### Header image
- **Type**: `string`
- **Description**: Header image URL in the store | Kapak resmi.

### Website
- **Type**: `string`
- **Description**: Website of the game | Oyunun resmi internet sitesi.

### Support url
- **Type**: `string`
- **Description**: Game support URL | Destek hattı internet adresi.

### Support email
- **Type**: `string`
- **Description**: Game support email | Destek hattı elektronik posta adresi.

### Windows
- **Type**: `bool`
- **Description**: Does it support Windows? | Windows destekliyor mu?

### Mac
- **Type**: `bool`
- **Description**: Does it support Mac OS? | Mac OS destekliyor mu?

### Linux
- **Type**: `bool`
- **Description**: Does it support Linux? | Linux destekliyor mu?

### Metacritic score
- **Type**: `int`
- **Description**: MetaCritic score, 0 if it has none | MetaCritic puanı, eğer yoksa 0.

### Metacritic url
- **Type**: `string`
- **Description**: MetaCritic review URL | MetaCritic yorum sayfa adresi.

### User score
- **Type**: `int`
- **Description**: Users score, 0 if it has none | Kullanıcı puanı, eğer yoksa 0.

### Positive
- **Type**: `int`
- **Description**: Positive votes | Olumlu oy sayısı.

### Negative
- **Type**: `int`
- **Description**: Negative votes | Olumsuz oy sayısı.

### Score rank
- **Type**: `string`
- **Description**: Score rank of the game based on user reviews | Kullanıcı değerlendirmesine göre sıralama puanı.

### Achievements
- **Type**: `int`
- **Description**: Number of achievements, 0 if it has none | Oyun içi başarı sayısı, eğer yoksa 0.

### Recommendations
- **Type**: `int`
- **Description**: User recommendation count | Tavsiye eden sayısı.

### Notes
- **Type**: `string`
- **Description**: Extra information about the game content | Oyun hakkında ek bilgiler.

### Average playtime forever
- **Type**: `int`
- **Description**: Average playtime since March 2009, in minutes | Mart 2009'dan itibaren ortalama oynanma dakikası.

### Average playtime two weeks
- **Type**: `int`
- **Description**: Average playtime in the last two weeks, in minutes | Son iki haftalık ortalama oynanma dakikası.

### Median playtime forever
- **Type**: `int`
- **Description**: Median playtime since March 2009, in minutes | Mart 2009'dan itibaren median oynanma dakikası.

### Median playtime two weeks
- **Type**: `int`
- **Description**: Median playtime in the last two weeks, in minutes | Son iki haftalık median oynanma dakikası.

### Developers
- **Type**: `string`
- **Description**: Developer name | Geliştiricilerinin ismi.

### Publishers
- **Type**: `string`
- **Description**: Publisher name | Sunan ismi.

### Categories
- **Type**: `string`
- **Description**: Category names | Kategori ve içerdiği özellikler örn: tek kişilik, başarı içeriyor vs.

### Genres
- **Type**: `string`
- **Description**: Genre names | Tür ismi örn: Aksiyon, Spor, Strateji.

  **IMPORTANT**: Fill empties with tags. | ÖNEMLİ: Boşlukları "tags" ile doldur.

### Tags
- **Type**: `string`
- **Description**: Tags | Oyuna uygun etiketler.

  **IMPORTANT**: Fill empties with genres. | ÖNEMLİ: Boşlukları "genres" ile doldur.

### Screenshots
- **Type**: `string`
- **Description**: Game screenshot URL | Oyun ekran görüntüsü sayfa adresi.

### Movies
- **Type**: `string`
- **Description**: Game movie URL | Oyun tanıtım filmi sayfa adresi.

### **TODO**: HTML etiket için tarama yap, varsa temizle.
### **TODO**: Genres ve Tags birlestirme konusunda karar ver, gerekirse tek ve daha zengin tek bir degisken haline getir.
