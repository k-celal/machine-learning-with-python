# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:17:21 2023

@author: Celal
"""

import pandas as pd
#internetten veri çekme

url= "https://bilkav.com/satislar.csv"
#veri onisleme
data= pd.read_csv(url)
data=data.values
X=data[:,0:1]
Y=data[:,1]

#verileri bolme
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

# from sklearn.linear_model import LinearRegression

# lr= LinearRegression()

# lr.fit(x_train,y_train)

# print(lr.predict(x_test))

#verinin kaydedilmesi
import pickle as pck

# dosya="model.mx"
# pck.dump(lr,open(dosya,'wb'))

#veri yükleme

yuklenen_model = pck.load(open("model.mx",'rb'))
y_pred=yuklenen_model.predict(x_test)
print(y_pred)