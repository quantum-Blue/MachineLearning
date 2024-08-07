import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/poly.csv")

x=data.zaman.values.reshape(-1,1)
y=data.sicaklik.values.reshape(-1,1)

from sklearn.preprocessing  import StandardScaler
sc=StandardScaler()

x1=sc.fit_transform(x)
y1=sc.fit_transform(y)

from sklearn.svm import SVR
# sv=SVR(kernel="rbf")
# sv=SVR(kernel="poly")
# sv=SVR(kernel="sigmoid")
sv=SVR(kernel="linear")

sv.fit(x1,y1)
plt.scatter(x1,y1)
plt.plot(x1,sv.predict(x1),color='red')
plt.title('Sigorta Sıcaklığı vs Zaman')
plt.xlabel('Zaman')
plt.ylabel('Sıcaklık')
plt.show()
