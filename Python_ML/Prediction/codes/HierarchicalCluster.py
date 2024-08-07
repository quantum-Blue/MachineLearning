import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/musteri.csv")
#print(data.head)

x=data.iloc[:,[3,4]].values

# cluster grafiği için

#import scipy.cluster.hierarchy as sch
#den=sch.dendrogram(sch.linkage(x,method="ward"))
#plt.show()

from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=5)
y_head=ac.fit_predict(x)

plt.scatter(x[y_head==0,0],x[y_head==0,1],color='red')
plt.scatter(x[y_head==1,0],x[y_head==1,1],color='green')
plt.scatter(x[y_head==2,0],x[y_head==2,1],color='blue')
plt.scatter(x[y_head==3,0],x[y_head==3,1],color='yellow')
plt.scatter(x[y_head==4,0],x[y_head==4,1],color='cyan')

plt.show()
