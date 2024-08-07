# NLP Doğal Dil İşleme, Natural Language Processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/twitter.csv",encoding="latin1")
#print(data)
data1=pd.concat([data.gender,data.description],axis=1)

data1.dropna(inplace=True)
print(data.head())
data1.reset_index(drop=True, inplace=True)

data1=pd.concat([data1.gender,data.description],axis=1)
data1.gender=[1 if i =="female" else 0 for i in data.gender]
print(data.head())

import re
metin=re.sub("[^a-zA-Z]"," ",data1.description[9])
print(metin) # noktalama işaretleri silinir yerine boşluk konur
harfler=metin.lower()
bol=harfler.split()

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

import nltk
nltk.download("stopwords")
stop=nltk.download("stopwords")

from nltk.corpus import stopwords
metin=[ps.stem(i) for i in bol if not i in set(stopwords.words("english"))]
metinson=" ".join(metin)

liste = []
for j in range(1000):
    metin=re.sub("[^a-zA-Z]" , " " , data1.description[j])
    metin=metin.lower()
    metin=metin.split()
    metin=[ps.stem(i) for i in metin if not i in set(stopwords.words("english"))]
    metinson=" ".join(metin)
    liste.append(metinson)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(10000)#.fit_transform(liste)
x=cv.fit_transform(liste).toarray()
y=data1.iloc[:10000,0].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

from sklearn.naive_bayes import GaussianNB
gn=GaussianNB()
gn.fit(x_train,y_train)
yhead=gn.predict(x_test) #([cv.transform([metin]).toarray()])

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,yhead)
print(cm)

print(gn.score(yhead,y_test))
#print(gn.score(yhead.reshape(-1,1),y_test))
