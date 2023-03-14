#import bolumu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yukleme
veriler = pd.read_csv("veriler.csv")
#veri on isleme
print(veriler)

boy =veriler[["boy"]]

print(boy)

boykilo=veriler[["boy","kilo"]]

print(boykilo)


