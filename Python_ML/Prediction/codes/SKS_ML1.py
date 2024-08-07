import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/maas.csv")
# data.head()
x=data.iloc[:,0].values.reshape(-1,1)
y=data.iloc[:,1].values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("Tecrube")
plt.ylabel("Maas")
# plt.show()

from sklearn.model_selection import train_test_split
xtrain,  xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=33)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(xtrain,ytrain)

# tahmin algoritması
# predict = tahmin
yhead = lr.predict(xtest)
lr.predict([[1.8]]) # 1.8

# plt.scatter(x,y)
# plt.plot(xtest, lr.predict(xtest), color="red")
# plt.show()

plt.scatter(x,y)
plt.plot(x, lr.predict(x), color="red")
plt.show()
