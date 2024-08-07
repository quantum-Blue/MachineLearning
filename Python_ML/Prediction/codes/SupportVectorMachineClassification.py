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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=23)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train1=sc.fit_transform(x_train)
x_test1=sc.transform(x_test)

from  sklearn.svm import SVC
#clf=SVC(kernel='linear') #rbf kernel ile da denemedik
sv=SVC(random_state=54)
sv.fit(x_train1,y_train.ravel())

yhead=sv.predict(x_test1)
sv.score(x_test1,y_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.ravel(),yhead)


sv1=SVC()
sv1.fit(x_train1,y_train.ravel())

yhead=sv1.predict(x_test1)
sv1.score(x_test1,y_test)
