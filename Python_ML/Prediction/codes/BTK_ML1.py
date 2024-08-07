
# Serhat Kağan Şahin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

plt.scatter([1,2,3,4,5],[1,8,27,64,125])
plt.xlabel("sayilar")
plt.ylabel("kupu")
plt.show()

plt.plot([1,2,3,4,5],[1,8,27,64,125])
plt.xlabel("sayilar")
plt.ylabel("kupu")
plt.show()


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=22)
print(len(xtrain), len(xtest))
print(xtrain)

xnorm=(x-np.min(x)/(np.max(x)-np.min(x)))  #normalize etmek için kullanılabilir
print(xnorm)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain1 = sc.fit_transform(xtrain)
"""

#       G P T
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error

# Veriyi yükle
data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/BTK.csv", header=None)

# Özellikleri (X) ve hedef değişkeni (Y) ayır
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1]

# Veri setini Eğitim ve Test setlerine ayır
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Özellik ölçekleme
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# VotingRegressor kullanarak Birden Fazla Regresyon Modelini Eğitme
regressor = VotingRegressor([('linear', LinearRegression()),
                             ('tree', DecisionTreeRegressor()),
                             ('forest', RandomForestRegressor())])

regressor.fit(X_train, Y_train)

# Test seti üzerinde tahmin yapma
Y_pred = regressor.predict(X_test)

# Modeli değerlendirme (regresyon görevleri için genellikle Hata Kare Ortalaması gibi metrikler kullanılır)
mse = mean_squared_error(Y_test, Y_pred)
print(f'Hata Kare Ortalaması: {mse}')
"""

# * * * * * * * blackbox * * * * * * * * * * * * * 

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data=pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/BTK.csv",header=None)
print(data)

X=data.iloc[:,:-1].values
Y=data.iloc[:,-1]

# print(X)
# print(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print(X_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import MultiLinearRegressor
regressor = MultiLinearRegressor([LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()])
regressor.fit(X_train, Y_train)

# Predicting on the Test set
Y_pred = regressor.predict(X_test)

# Creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

print(cm)

"""

################ BTK

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/veriler.csv", header=None)
print(data)

boy=data[["boy"]]
print(boy)

boykilo = data["boy","kilo"]
print(boykilo)

class insan:
    boy=180
    def kosmak(self,b):
        return b + 10
    
ali = insan()
print(ali.boy)
print(ali.kosmak(90))

l=[1,3,4] # liste
"""

#############  devamke (ÇALIŞMADI BU DA)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

veri = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/eksikveriler.csv", header=None)

from sklearn.impute  import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
yas = veri.iloc["yas"].values
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[ : , 1:4])
print(yas)

from sklearn import preprocessing

ulke = veri.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
ulke[:,0:1] = le.fit_transform(veri.iloc[:,0:1])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
"""

# # # # # # # SONUNDA ÇALIŞIYORR

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

# Veriyi yükle
veri = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/eksikveriler.csv", header=None)

# 'yas' sütununu seç
yas = veri.iloc[:, 3:4].values

# Eksik değerleri en sık rastlanan değerle doldur
imputer = SimpleImputer(strategy="most_frequent")
yas = imputer.fit_transform(yas)

print("Eksik Değerleri Doldurulmuş 'yas' Sütunu:")
print(yas)

# 'ulke' sütununu Label Encoding ve One-Hot Encoding ile işle
ulke = veri.iloc[:, 0:1].values
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veri.iloc[:, 0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

print("Label Encoding ve One-Hot Encoding Uygulanmış 'ulke' Sütunu:")
print(ulke)


#########

print(list(range(22)))

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2=pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet=veri.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)
