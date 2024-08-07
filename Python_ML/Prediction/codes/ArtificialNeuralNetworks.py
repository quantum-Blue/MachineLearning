"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)
data=pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/Churn_Modelling.csv")
#print(data.head())

x=data.iloc[:,3:-1].values
y=data.iloc[:,:-1].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[1])],remainder="passthrough")
#ct=ColumnTransformer([('ascii', 'utf-8')], remainder='passthrough')  #remainder is for the columns which we don't want to transform
x=np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation="relu")) #gizli katman1 
ann.add(tf.keras.layers.Dense(units=6,activation="relu")) #gizli katman2
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid")) #çıkış katmanı
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
ann.fit(x_train,y_train,epochs=100,verbose=1)
#ann.fit(x_train,y_train,epochs=100,batch_size=32)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)
data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/Churn_Modelling.csv")

x = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values  # 'Exited' sütununu seç

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))  # gizli katman1
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))  # gizli katman2
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))  # çıkış katmanı
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
ann.fit(x_train, y_train, epochs=100)

ypred=ann.predict(x_test)
ypred=(ypred>0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,ypred)
print(f"Accuracy: {accuracy_score(y_test,ypred)}")
print(f"Confusion Matrix:\n{cm}")
