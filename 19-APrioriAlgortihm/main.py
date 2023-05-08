# -*- coding: utf-8 -*-
"""
Created on Tue May  9 00:58:55 2023

@author: Celal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler= pd.read_csv("sepet.csv",header=None)

t=[]

for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])

#https://github.com/ymoch/apyori
from apyori import apriori
kurallar= apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_lenght=2)
print(list(kurallar))