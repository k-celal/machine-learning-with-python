# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 23:29:58 2023

@author: Celal
"""

#1)Gerekli kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2)veri onisleme
#2.1)Verilerin yuklenmesi
veriler = pd.read_csv("eksikveriler.csv")

#3)Eksik veriler

# df= pd.DataFrame(veriler)
# print(df.describe())
# df=df.fillna(28)
# print(df)

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)
#4)encoder kategorik veriler -> numeric veri

ulke=veriler.iloc[:,0:1].values
print(ulke)
from sklearn import preprocessing

le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

#kolonları 1 veya 0 atamak
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#numpy dizileri dataframe donusumu
sonuc= pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

#verileri birleştirme
s=pd.concat([sonuc,sonuc2],axis=1) #axis 1 demek yanyana ekleme axis=0 demek alt alta ekleme
s2=pd.concat([s,sonuc3],axis=1)
print(s)
#verilerin bolunmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)
#sayısal degerlerin ayni dünyaya nasıl getirilecek - olcekleme standartlaşma-normalizasyon
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

