import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

data = pd.read_csv('/Users/enesbal/Desktop/m_learn/csv_files/Avm_Musterileri.csv')

plt.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Correlation between Annual income and Spending score")
plt.show()

scaler = MinMaxScaler()
scaler.fit(data[["Annual Income (k$)"]])
data["Annual Income (k$)"] = scaler.transform(data[["Annual Income (k$)"]])

scaler.fit(data[["Spending Score (1-100)"]])
data["Spending Score (1-100)"] = scaler.transform(data[["Spending Score (1-100)"]])

dirsek = range(1, 11)
liste = []
for k in dirsek:
    km = KMeans(n_clusters=k)
    km.fit(data[["Annual Income (k$)", "Spending Score (1-100)"]])
    liste.append(km.inertia_)

plt.xlabel("k")
plt.ylabel("dirsek")
plt.plot(dirsek, liste)
plt.show()

kson = KMeans(n_clusters=5)
y_pred = kson.fit_predict(data[["Annual Income (k$)", "Spending Score (1-100)"]])

data["cluster"] = y_pred
centroids = kson.cluster_centers_

data1 = data[data["cluster"] == 0]
data2 = data[data["cluster"] == 1]
data3 = data[data["cluster"] == 2]
data4 = data[data["cluster"] == 3]
data5 = data[data["cluster"] == 4]

plt.xlabel("gelir")
plt.ylabel("skor")

plt.scatter(data1["Annual Income (k$)"], data1["Spending Score (1-100)"], color="red")
plt.scatter(data2["Annual Income (k$)"], data2["Spending Score (1-100)"], color="blue")
plt.scatter(data3["Annual Income (k$)"], data3["Spending Score (1-100)"], color="green")
plt.scatter(data4["Annual Income (k$)"], data4["Spending Score (1-100)"], color="teal")
plt.scatter(data5["Annual Income (k$)"], data5["Spending Score (1-100)"], color="pink")

plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x", label="centroid")
plt.legend()
plt.show()
