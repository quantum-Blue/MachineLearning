import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/diabetes.csv")

sekerHastasi=data[data.Outcome==1]
saglikli=data[data.Outcome==0]

plt.scatter(saglikli.Age,saglikli.Glucose,color=['green'],label='Saglikli',alpha=0.4)
plt.scatter(sekerHastasi.Age,sekerHastasi.Glucose,color=['red'],label='Hasta',alpha=0.4)
plt.xlabel("yas")
plt.ylabel("glukoz")
plt.show()

x1=data.iloc[:,0:-1]
y1=data.iloc[:,-1].values

from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test=train_test_split(x1, y1,test_size=0.3, random_state=43)

from sklearn.preprocessing import  StandardScaler
sc=StandardScaler()
x_train1=sc.fit_transform(x_train)
x_test1=sc.transform(x_test)

from sklearn.neighbors import  KNeighborsClassifier
knn=KNeighborsClassifier(12)
knn.fit(x_train1,y_train)
KNeighborsClassifier(n_neighbors=12)
yhead=knn.predict(x_test1)
knn.score(x_test1,y_test)

scorelite = []
for i in range(1, 30):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train1, y_train)
    scorelite.append(knn2.score(x_test1, y_test))

plt.plot(range(1, 30), scorelite)
plt.xlabel("komsu sayisi")
plt.ylabel("dogruluk orani")
plt.show()

from sklearn.metrics import confusion_matrix
cm=confusion_matrix()
print(cm)
