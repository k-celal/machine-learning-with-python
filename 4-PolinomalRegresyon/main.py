# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:22:35 2023

@author: Celal
"""


#1)Gerekli kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2)veri onisleme
#2.1)Verilerin yuklenmesi
veriler = pd.read_csv("maaslar.csv")


x=veriler.iloc[:,1:2].values
y=veriler.iloc[:,2:].values

#lineer regresyon

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x, y)
plt.scatter(x, y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.show()
#polinomal regresyon

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
