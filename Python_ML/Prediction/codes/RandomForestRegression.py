import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/bilet.csv")

x=data.sira.values.reshape(-1,1)
y=data.fiyat.values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=22) # n_estimators: ağaç sayısı
rf.fit(x,y)
yhead=rf.predict(x) # ara değer de çıkabliyor
plt.scatter(x,y,color="red")
plt.plot(x,yhead)
plt.show()

x1=np.arange(min(x),max(x),0.01).reshape(-1,1)
yhead2=rf.predict(x1)

#   R Square Error Teorisi

from sklearn.metrics import r2_score

r2_score(y,yhead)


