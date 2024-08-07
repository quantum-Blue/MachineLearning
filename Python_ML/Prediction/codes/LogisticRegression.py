import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/urun.csv")

x=data.iloc[:,0:2].values
y=data.satinalma.values.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=44)

S=data[data.satinalma==0]
B=data[data.satinalma==1]
plt.scatter(S.yas,S.maas,color="red")
plt.scatter(B.yas,B.maas,color="green")

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train1=sc.fit_transform(x_train)
x_test1=sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train1,y_train)

yhead=lr.predict(x_test1)
lr.score(x_test1,y_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,yhead)

plt.show()
