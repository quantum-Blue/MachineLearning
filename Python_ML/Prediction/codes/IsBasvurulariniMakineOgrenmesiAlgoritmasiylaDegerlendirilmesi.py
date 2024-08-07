import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/DecisionTreesClassificationDataSet.csv",names=["sepal length","sepal weight","petal lenght","petal height","target"])

d={"Y":1,"N":0}

data["IseAlindi"]=data["IseAlindi"].map(d)
data["SuanCalisiyor?"]=data["SuanCalisiyor?"].map(d)
data["Top10 Universite?"]=data["Top10 Universite?"].map(d)
data["StajBizdeYaptimi?"]=data["StajBizdeYaptimi?"].map(d)

e={"BS":1,"PhD":2,"MS":3}
data["EgitimSeviyesi"]=data["EgitimSeviyesi"].map(e)

x=data.iloc[:,0:-1]
y=data.iloc[:,-1].values

from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier()
dc.fit(x,y)
#print(dc.feature_importances_)
dc.predict([[4,1,3,1,0,1]])
