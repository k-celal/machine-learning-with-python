# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:25:25 2023

@author: Celal
"""

#1)Gerekli kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2)veri onisleme
#2.1)Verilerin yuklenmesi
veriler = pd.read_csv("veriler.csv")

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
c=veriler.iloc[:,-1:].values
print(c)
from sklearn import preprocessing

le=preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(c)

#kolonları 1 veya 0 atamak
ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

#numpy dizileri dataframe donusumu
sonuc= pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)
sonuc3=pd.DataFrame(data= c[:,:1],index=range(22),columns=['cinsiyet'])
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

#modelin egitilmesi

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

#boy tahmin
boy=s2.iloc[:,3:4].values

sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]
veri=pd.concat([sol,sag],axis=1)

x_train,x_test,y_train,y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression
r2=LinearRegression()
r2.fit(x_train,y_train)

y_pred=r2.predict(x_test)

#modelin başarısını ölcmek

import statsmodels.api as sm
#y=a+bx+cx formülündeki a=1 olarak eklendi
X = np.append(arr=np.ones((22,1)).astype(int), values=veri,axis=1)
#backward elimination

X_l=veri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy, X_l).fit()
print(model.summary())

X_l=veri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy, X_l).fit()
print(model.summary())

X_l=veri.iloc[:,[0,1,2,3]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy, X_l).fit()
print(model.summary())


