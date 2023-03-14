
"""
Spyder Editor

This is a temporary script file.
"""


#1)Gerekli kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2)veri onisleme
#2.1)Verilerin yuklenmesi
veriler = pd.read_csv("satislar.csv")

aylar = veriler[['Aylar']]
satislar= veriler[['Satislar']]


# verilerin bolunmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)
# #sayısal degerlerin ayni dünyaya nasıl getirilecek - olcekleme standartlaşma-normalizasyon
# from sklearn.preprocessing import StandardScaler

# sc=StandardScaler()
# X_train=sc.fit_transform(x_train)
# X_test=sc.fit_transform(x_test)
# Y_train=sc.fit_transform(y_train)
# Y_test=sc.fit_transform(y_test)
# #dogrusal model regresyon model insası

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)
tahminsonuc =lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara gore satis")