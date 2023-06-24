# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 23:59:18 2023

@author: Celal
"""


#1)Gerekli kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2)veri onisleme
#2.1)Verilerin yuklenmesi
veriler = pd.read_csv("Wine.csv")
print(veriler)
#rownumber , customerid,surname gereksiz sütunlar
X=veriler.iloc[:,0:13].values
Y=veriler.iloc[:,13].values

#verilerin bolunmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#sayısal degerlerin ayni dünyaya nasıl getirilecek - olcekleme standartlaşma-normalizasyon
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
#tek boyut verdik çünkü hangi sınıfta olduğu önemli değil verinin
X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

#pca dönüşümünden önce logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#pca dönüşümünden sonra logistic regression
classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

#tahminler
y_pred=classifier.predict(X_test)

y_pred2= classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
#actual Pca olmadan çıkan sonuç
print("Gercek ve PCA olmadan")
cm=confusion_matrix(y_test,y_pred)
print(cm)
#actual Pca sonrası çıkan sonuç
print("Gercek ve PCA sonrası")
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)
#actual PCAönce/PcaSonra
print("PCA olmadan ve PCA lı")
cm3=confusion_matrix(y_pred,y_pred2)
print(cm2)

#LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda= LDA(n_components=2)
#y_train vermeliyiz çünkü verinin hangi sınıfa ait olduğu önemli
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)

#LDA dönüşümünden sonra logistic regression
classifier_lda=LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)

#tahminler
y_pred_lda= classifier_lda.predict(X_test_lda)

from sklearn.metrics import confusion_matrix
#actual LDA olmadan çıkan sonuç
print("Gercek ve LDA olmadan")
cm4=confusion_matrix(y_test,y_pred)
print(cm4)
#actual Pca sonrası çıkan sonuç
print("Gercek ve LDA sonrası")
cm5=confusion_matrix(y_test,y_pred_lda)
print(cm5)
#actual PCAönce/PcaSonra
print("LDA olmadan ve LDA lı")
cm6=confusion_matrix(y_pred,y_pred_lda)
print(cm6)



