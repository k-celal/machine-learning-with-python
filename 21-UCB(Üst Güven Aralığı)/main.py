# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:47:15 2023

@author: Celal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler =pd.read_csv("Ads_CTR_Optimisation.csv")
# RandomSelection
import random
import math
N=10000
d=10
toplam=0
secilenler=[]
for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=veriler.values[n,ad] #verilerdeki n.satır ad. sutun 1 ise odul 1 oluyor toplama ekleniyor
    toplam=toplam+odul

print("Toplam odul RastgeleSecim")
print(toplam)
    
plt.hist(secilenler)
plt.show()

# UCB

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
            delta=math.sqrt((3/2)*(math.log(n)/tiklamalar[i]))
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
print("Toplam Odul UCB")
print(toplam)

plt.hist(secilenler)
plt.show()

#ThompsonÖrnekleme

N=10000 #10000 tıklama
d=10 #10 ilan var
toplam=0 #toplam odul
secilenler=[]
birler=[0]*d
sifirlar=[0]*d
for n  in range(0,N):
    ad=0#secilen ilan
    max_th=0
    for i in range(0,d):
        rasbeta=random.betavariate(birler[i]+1, sifirlar[i]+1)
        if rasbeta>max_th:
            max_th=rasbeta
            ad=i
    secilenler.append(ad)
    odul=veriler.values[n,ad] #verilerdeki n.satır ad. sutun 1 ise odul 1 oluyor toplama ekleniyor
    if odul==1:
        birler[ad]=birler[ad]+1
    else:
        sifirlar[ad]= sifirlar[ad]+1
    toplam=toplam+odul
print("Toplam Odul Thompson")
print(toplam)

plt.hist(secilenler)
plt.show()