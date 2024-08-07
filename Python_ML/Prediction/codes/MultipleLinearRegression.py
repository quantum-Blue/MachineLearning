import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/reklam.csv")

# regresyon = b0 + b1 + x
# maas = b0 + b1(eğim) + tecrube

x=data.iloc[:,1:4].values
y=data.satis.values.reshape(-1,1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=22)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
# lr.fit(xtrain,ytrain)
lr.fit(x,y)

yhead=lr.predict(xtest)
# lr.predict([[xtest]])
lr.predict([[230,38,70]])