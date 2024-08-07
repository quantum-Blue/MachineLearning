import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/pca_iris.data",names=["sepal length","sepal weight","petal lenght","petal height","target"])

from sklearn import linear_model
ml=linear_model.LinearRegression()

x=data.iloc[:,0:-1]
y=data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x1=sc.fit_transform(x)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
princpialComponent=pca.fit_transform(x1)
princpialDF=pd.DataFrame(data=princpialComponent,columns=["p1","p2"])
print(princpialDF.head)

son=pd.concat([princpialDF,data[["target"]]],axis=1)
print(son)

setosa=son[data.target=="Iris-setosa"]
virginica=son[data.target=="Iris-virginica"]
versicolor=son[data.target=="Iris-versicolor"]

plt.xlabel("p1")
plt.xlabel("p2")
plt.scatter(setosa["p1"],setosa["p2"],color="blue")
plt.scatter(virginica["p1"],virginica["p2"],color="green")
plt.scatter(versicolor["p1"],versicolor["p2"],color="red")
plt.show()
