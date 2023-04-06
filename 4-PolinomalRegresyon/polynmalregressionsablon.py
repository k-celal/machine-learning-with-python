# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:22:35 2023

@author: Celal
"""


#1)Gerekli kutuphaneler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#2)veri onisleme
#2.1)Verilerin yuklenmesi
veriler = pd.read_csv("maaslar.csv")

#dataframe dilimleme
X=veriler.iloc[:,1:2]
Y=veriler.iloc[:,2:]
x=X.values
y=Y.values

#lineer regresyon doğrusal model

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x, y)

#2.derece polinomal regresyon doğrusal olmayan model

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#4.dereceden deneme kısmı


poly_reg3 = PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(x)

lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,y)

#gorsellestirme

plt.scatter(x, y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(x)),color='blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
