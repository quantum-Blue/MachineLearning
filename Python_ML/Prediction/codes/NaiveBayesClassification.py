import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/urun.csv")

x=data.iloc[:,0:2].values
y=data.satinalma.values.reshape(-1,1)

S=data[data.satinalma==0]
B=data[data.satinalma==1]

plt.scatter(S.yas,S.maas,color="red")
plt.scatter(B.yas,B.maas,color="green")

from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, random_state=23)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train1=sc.fit_transform(x_train)
x_test1=sc.transform(x_test)

#   GaussianNB : Tahmin edeceğimiz veri ya da kolon sürekli reel veya kategorik olup olmadığını belirler.
#   BernoulliNB : Tahmin edeceğimiz veri (ikili ise) bir kategorik değeri içerdiğinde kullanılır
#   MultinomialNB : Tahmin edeceğimiz veri integer sayilardan oluşuyorsa
#(Çoklu labeled classlar varsa kullanılır.)

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB() # Yeni bir n baylonlu ve çözüm oluşturmak için Gaussian NB'yi kullanacağız.
nb.fit(x_train1,y_train.raven())
yhead=nb.predict(x_train1)
nb.score(x_test1,y_test)


from sklearn.naive_bayes import BernoulliNB
nb=BernoulliNB()
nb.fit(x_train1,y_train.raven())
yhead=nb.predict(x_train1)
nb.score(x_test1,y_test)

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(x_train1,y_train.raven())
yhead=nb.predict(x_train1)
nb.score(x_test1,y_test)

