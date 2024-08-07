# GPT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor  # KNeighborsRegressor eklenmiş
from sklearn.model_selection import cross_val_score

data = pd.read_csv('/Users/enesbal/Desktop/m_learn/csv_files/Avm_Musterileri.csv')

plt.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Correlation between Annual income and Spending score")
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Normalizing the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[['Annual Income (k$)', 'Spending Score (1-100)']])
df_scaled = pd.DataFrame(scaled_features, columns=['Annual Income', 'Spending Score'])
print(df_scaled.head())

# Creating the DataFrame with the target variable
target_var = df_scaled.iloc[:, -1]  # Target Variable is the last column in the dataframe
df_num = df_scaled.drop('Spending Score', axis=1)  # Removing the Target Variable from the Num Features

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df_num, target_var, test_size=0.2, random_state=42)

# Finding the best value of K for KNN algorithm using cross validation method
knn = KNeighborsRegressor()  # KNeighborsRegressor kullanılıyor
scores = cross_val_score(knn, X_train, y_train, cv=5)
avg_accuracy = scores.mean()
print("The average accuracy is: ", avg_accuracy)

# Using different values of k to find the optimal one
k_values = range(1, 31)
cv_scores = []
for i in k_values:
    knn.n_neighbors = i
    cv_scores.append(cross_val_score(knn, X_train, y_train, cv=5).mean())

optimal_k = k_values[cv_scores.index(max(cv_scores))]
print("The optimal number of neighbours is:", optimal_k)


# BLACKBOX
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.preprocessing import MinMaxScaler

data=pd.read_csv('/Users/enesbal/Desktop/m_learn/csv_files/Avm_Musterileri.csv')

plt.scaler(data["Annual Income (k$)"],data["Spending Score (1-100)"])
plt.xlabel(("Annual Income"))
plt.ylabel(("Spending Score"))
plt.title(("Correlation between Annual income and Spending score"))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

#Normalizing the Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_features = scaler.fit_transform(data[['Annual Income (k$)','Spending Score (1-100)']])
df_scaled = pd.DataFrame(scaled_features, columns=['Annual Income', 'Spending Score'])
print(df_scaled.head())

#Creating the DataFrame with the target variable
target_var = df_scaled.iloc[:,-1] #Target Variable is the last column in the dataframe
df_num = df_scaled.drop('Yıllık Gelir', axis=1) #Removing the Target Variable from the Num Features

#Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_num, target_var, test_size=0.2, random_state=42)

#Finding the best value of K for KNN algorithm using cross validation method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier()
scores = cross_val_score(knn, X_train, y_train, cv=5)
avg_accuracy = scores.mean()
print ("The average accuracy is: ", avg_accuracy)

#Using different values of k to find the optimal one
k_values = range(1,31)
cv_scores = []
for i in k_values:
    knn.n_neighbors = i
    cv_scores.append(cross_val_score(knn, X_train, y_train, cv=5))
max_acc = max([item for sublist in cv_scores for item in sublist])
optimal_k = k_values[cv_scores.index(cv_scores.tolist().index(max_acc))] 
print("The optimal number of neighbours is :", optimal_k+1) 
"""