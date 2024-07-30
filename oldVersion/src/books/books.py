import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


#1
users = '/Users/avokado/PycharmProjects/miuul/Final/Users.csv'
df1 = pd.read_csv(users, sep=',')
df1.head()
df1.shape
# istersek; Lokasyonlari ayirmak-Lokasyonlara göre popülerlik-enler gibi seylere bakilabilir.
df1_expanded = df1.assign(Location=df1['Location'].str.split(',')).explode('Location')


#2
rat_b ='/Users/avokado/PycharmProjects/miuul/Final/Ratings.csv'
df2 = pd.read_csv(rat_b, sep=',')
df2.head()
df2.shape



#3
book ='/Users/avokado/PycharmProjects/miuul/Final/Books.csv'
df3 = pd.read_csv(book , sep=',')
df3.head()
df3.shape



# İki veri seti (rat_b + book) - Kitaba(ISBN) göre ve tüm sütunlar.
merged_df = pd.merge(df2, df3 , on='ISBN', how='inner')
merged_df.head()
merged_df.shape


# Istersek
#sadece mesela iki sütun (Book Title ve Boot Rating )
merged_df = df2.merge(df3, on='ISBN')
# Kitaplara göre ratingler count olarak(ortalamalarinida hesaplayip ikisiyle bir derece elde edebiliriz)
merged_df_count= merged_df.groupby('Book-Title').count()['Book-Rating'].reset_index()
merged_df_count .head()
# Popülerlik - benzerlik - en cok - en az  begenilen gibi gibi hesaplamalar cogalabilir.


#3. veri setini de ekleme (users + rat_b +books) (karar vermeliyiz - bilemedim -gerek yok giib cünkü userslar gizli)
final_merged_df = pd.merge(merged_df, df1, on='?', how='inner') # bu kod calistirilmamistir.
final_merged_df.head()


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

check_detail(dataframe=merged_df)

