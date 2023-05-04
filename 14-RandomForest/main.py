# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:49:50 2023

@author: Celal
"""


#1)Gerekli kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2)veri onisleme
#2.1)Verilerin yuklenmesi
veriler = pd.read_csv("veriler.csv")
x=veriler.iloc[5:,1:4].values
y=veriler.iloc[5:,4:].values
print(veriler)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)
#sayısal degerlerin ayni dünyaya nasıl getirilecek - olcekleme standartlaşma-normalizasyon
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred=log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train, y_train)

knn_y_pred=knn.predict(X_test)

cm=confusion_matrix(y_test, knn_y_pred)
print(cm)

from sklearn.svm import SVC

svc=SVC(kernel='sigmoid')
svc.fit(X_train,y_train)
svc_y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test, svc_y_pred)
print(cm) 

#svm cekirdek hilesi
svc=SVC(kernel='poly')
svc.fit(X_train,y_train)
svc_y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test, svc_y_pred)
print(cm) 

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
print("Naive bayes")
cm=confusion_matrix(y_test, svc_y_pred)
print(cm)

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
print("Decisiontree")
cm=confusion_matrix(y_test, svc_y_pred)
print(cm)
from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier(n_estimators=10,criterion="gini")
RFC.fit(X_train,y_train)
y_pred=RFC.predict(X_test)
print("Random Forest")
cm=confusion_matrix(y_test, svc_y_pred)
print(cm)
