import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/bilet.csv")

x=data.sira.values.reshape(-1,1)
y=data.fiyat.values.reshape(-1,1)

plt.scatter(x,y)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x,y)
dt.predict([[3.5]])

yhead=dt.predict(x)
plt.scatter(x,y)
plt.plot(x,yhead)

x1=np.arange(min(x),max(x),0.1).reshape(-1,1)
yhead2=dt.predict(x1)

plt.scatter(x,y)
plt.plot(x1,yhead2)
plt.show()
