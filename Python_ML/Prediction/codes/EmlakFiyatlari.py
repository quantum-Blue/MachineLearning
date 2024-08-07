import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/multilinearregression.csv",sep=";")

from sklearn import linear_model
ml=linear_model.LinearRegression()

x=data.iloc[:,0:-1]
y=data.iloc[:,-1].values

ml.fit(x,y)

ml.predict([[250,4,2]]) # oda metrekaresi , oda sayısı , binanın yaşı
print(ml) # değerlerle oynayıp artıp azalmadığını kontrol edebilirsin

import pickle # yapay zekayı eğittikten sonra saklamak için
model="fiyat_tahmin.pickle"
pickle.dump(ml,open(model,"wb"))
# PİCKLE HATA VERİYO
