# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:47:15 2023

@author: Celal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler =pd.read_csv("Ads_CTR_Optimisation.csv")
#RandomSelection
import random

N=10000
d=10
toplam=0
secilenler=[]
for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=veriler.values[n,ad] #verilerdeki n.satır ad. sutun 1 ise odul 1 oluyor toplama ekleniyor
    toplam=toplam+odul
    
plt.hist(secilenler)
plt.show()