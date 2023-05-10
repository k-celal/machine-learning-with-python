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
# import random

# N=10000
# d=10
# toplam=0
# secilenler=[]
# for n in range(0,N):
#     ad=random.randrange(d)
#     secilenler.append(ad)
#     odul=veriler.values[n,ad] #verilerdeki n.satır ad. sutun 1 ise odul 1 oluyor toplama ekleniyor
#     toplam=toplam+odul
    
# plt.hist(secilenler)
# plt.show()

#UCB
import math
N=10000 #10000 tıklama
d=10 #10 ilan var
oduller=[0]*d #basta butun ilanların odulu 0 Ri(n)
toplam=0 #toplam odul
tiklamalar=[0]*d#o ana kadar tıklamalar Ni(n)
secilenler=[]
for n  in range(0,N):
    ad=0#secilen ilan
    max_ucb=0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama=oduller[i]/tiklamalar[i]
            delta=math.sqrt((1/100)*(math.log(n)/tiklamalar[i]))
            ucb=ortalama+delta
        else:
            ucb=N*10
        if max_ucb <ucb:#max tan buyuk bir ucb cıktı
            max_ucb=ucb
            ad=i
        
    secilenler.append(ad)
    tiklamalar[ad]=tiklamalar[ad]+1
    odul=veriler.values[n,ad] #verilerdeki n.satır ad. sutun 1 ise odul 1 oluyor toplama ekleniyor
    oduller[ad]=oduller[ad]+odul
    toplam=toplam+odul
print("Toplam Odul")
print(toplam)

plt.hist(secilenler)
plt.show()