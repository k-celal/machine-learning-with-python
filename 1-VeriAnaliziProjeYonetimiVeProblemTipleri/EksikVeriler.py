#import bolumu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yukleme
veriler = pd.read_csv("eksikveriler.csv")
print(veriler)

# df= pd.DataFrame(veriler)
# print(df.describe())
# df=df.fillna(28)
# print(df)
#veya
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)
