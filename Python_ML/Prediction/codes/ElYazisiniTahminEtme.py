import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.datasets import fetch_openml
mnist=fetch_openml("mnist_784")
def resim(dframe,index):
    numara=dframe.to_numpy()[index]
    numara_resim=numara.reshape(28,28)
    plt.imshow(numara_resim,cmap="binary")
    plt.axis("off")
    plt.show()

resim(mnist.data,2)

from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, y_test = train_test_split(mnist.data, mnist.target, test_size=1/7, random_state=33)

type(train_img)

test_img_kopya=test_img.copy()
resim(test_img_kopya,55)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(train_img)
train_img=sc.transform(train_img)
test_img=sc.transform(test_img)

print(mnist.data.shape)

from sklearn.decomposition import PCA
pca=PCA(.95)
pca.fit(train_img)
pca.n_components_

train_img=sc.transform(train_img)
test_img=sc.transform(test_img)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="lbfgs",max_iter=1000)
lr.fit(train_img,train_lbl)
lr.predict(test_img[66]).reshape(1,-1)
resim(test_img_kopya,66)

lr.predict(test_img[1248]).reshape(1,-1)
resim(test_img_kopya,1248)


