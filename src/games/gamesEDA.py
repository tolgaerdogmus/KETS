import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# AppID                             (string) Unique identifier for each app | Benzersiz oyun ID.
# Name                              (string) Game name | Oyun ismi.
# Release date                      (string) Release date of the game | Cikis tarihi.
# Estimated owners                  (string, e.g.: "0 - 20000") | Yaklasik oyuna sahip kisi sayisi.
# Peak CCU                          (int) Number of concurrent users, yesterday | Bir gün önceki aynı anda oyun oynayan kisi sayisi.
# Required age                      (int) Age required to play, 0 if it is for all audiences | Oynamak icin gereken yas siniri, 0 ise genele uygun.
# Price                             (float) Price in USD, 0.0 if its free | Fiyat USD, 0.0 ise ucretsiz.
# DLC count                         (int) Number of DLCs, 0 if you have none. | Ek icerik sayisi, 0 ise yok.
# About the game                    (string) Detailed description of the game | Detayli aciklama.
# Supported languages               (string, list) Supported languages | Desteklenen yazı dilleri.
# Full audio languages              (string, list) Languages with audio support | Desteklenen ses dilleri.
# Reviews                           (string) Reviews of the game | Yapilan elestiriler, yorumlar.
# Header image                      (string) Header image URL in the store | Kapak resmi.
# Website                           (string) Website of the game | Oyunun resmi internet sitesi.
# Support url                       (string) Game support URL | Destek hatti internet adresi.
# Support email                     (string) Game support email | Destek hatti elektronik posta adresi.
# Windows                           (bool)  Does it support Windows? | Windows destekliyor mu?
# Mac                               (bool)  Does it support Mac OS? | Mac OS destekliyor mu?
# Linux                             (bool)  Does it support Linux? | Linux destekliyor mu?
# Metacritic score                  (int) MetaCritic score, 0 if it has none | MetaCritic puani, eger yoksa 0.
# Metacritic url                    (string) MetaCritic review URL | MetaCritic yorum sayfa adresi.
# User score                        (int) Users score, 0 if it has none | Kullanici puani, eger yoksa 0.
# Positive                          (int) Positive votes | Olumlu oy sayisi.
# Negative                          (int) Negative votes | Olumsuz oy sayisi.
# Score rank                        (string) Score rank of the game based on user reviews | Kullanici degerlendirmesine gose siralama puani.
# Achievements                      (int) Number of achievements, 0 if it has none | Oyun ici basarim sayisi, eger yoksa 0.
# Recommendations                   (int) User recommendation count | Tavsiye eden sayisi.
# Notes                             (string) Extra information about the game content | Oyun hakkında ek bilgiler.
# Average playtime forever          (int) Average playtime since March 2009, in minutes | Mart 2009 dan itibaren ortalama oynanma dakikasi.
# Average playtime two weeks        (int) Average playtime in the last two weeks, in minutes | Son iki haftalik ortalama oynanma dakikasi.
# Median playtime forever           (int) Median playtime since March 2009, in minutes | Mart 2009 dan itibaren median oynanma dakikasi.
# Median playtime two weeks         (int) Median playtime in the last two weeks, in minutes | Son iki haftalik median oynanma dakikasi.
# Developers                        (string) Developer name | Gelistiricilerinin ismi.
# Publishers                        (string) Publisher name | Sunan ismi.
# Categories                        (string) Category names | Kategori ve icerdigi ozellikler orn: tek kisilik, basarim iceriyor vs.
# Genres                            (string) Genre names | Tur ismi orn: Aksiyon, Spor, Strateji.
##                                  IMPORTANT, fill empties with tags. | ONEMLI, bosluklari tags ile doldur.
# Tags                              (string) Tags | Oyuna uygun etiketler.
##                                  IMPORTANT, fill empties with genres. | ONEMLI, bosluklari genres ile doldur.
# Screenshots                       (string) Game screenshot URL | Oyun ekran goruntusu sayfa adresi.
# Movies                            (string) Game movie URL | Oyun tanitim filmi sayfa adresi.
# TODO: HTML etiket icin tarama yap varsa temizle.
# TODO: Genres ve Tags birlestirme konusunda karar ver, gerekirse tek ve daha zengin tek bir degisken haline getir

