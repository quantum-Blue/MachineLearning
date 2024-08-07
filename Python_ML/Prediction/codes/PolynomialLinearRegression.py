import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/poly.csv")

x = data.zaman.values.reshape(-1,1)
y = data.sicaklik.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)

plt.scatter(x,y)
plt.plot(x,lr.predict(x),color="red")

from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=6) # polinomun kaçıncı derece olduğunu gösterir
xpl=pr.fit_transform(x)

lr2=LinearRegression()
lr2.fit(xpl,y)

yhead2=lr2.predict(xpl)

plt.scatter(x,y)
plt.plot(x,lr.predict(x),color="red")
plt.plot(x,lr2.predict(xpl),color="blue")
plt.title("Degree 6 Poly Regression")
plt.show()

