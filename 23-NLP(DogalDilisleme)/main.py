# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:58:34 2023

@author: Celal
"""

import numpy as np
import pandas as pd
import nltk 
import re
#PREPROCESSİNG
yorumlar=pd.read_excel("Restaurant_Reviews.xlsx")

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.corpus import stopwords

duzenlenmis_yorum=[]
for i in range(996):
    yorum=re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum=yorum.lower()
    yorum=yorum.split()
    yorum=[ps.stem(kelime)for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum=' '.join(yorum)
    duzenlenmis_yorum.append(yorum)

#Feature Extraction(Oznitelik cıkarımı)
#Bag of words
from sklearn.feature_extraction.text import CountVectorizer

CV=CountVectorizer(max_features=3000)
#x bağımsız y bağımlı
X=CV.fit_transform(duzenlenmis_yorum).toarray()
y=yorumlar.iloc[:,1]

#makine ogrenmesi
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print("Naive Bayes")
print(cm)

from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier(n_estimators=10,criterion="gini")
RFC.fit(X_train,y_train)
y_pred=RFC.predict(X_test)
print("Random Forest")
cm=confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train, y_train)

knn_y_pred=knn.predict(X_test)

cm=confusion_matrix(y_test, knn_y_pred)
print("KNN")
print(cm)