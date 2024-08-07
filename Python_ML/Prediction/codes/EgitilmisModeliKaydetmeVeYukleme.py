import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model

import pickle

myModel=pickle.load(open("fiyat_tahmin.pickle","rb"))  # load the saved model

myModel.predict([[250,4,2]])
# PİCKLE HATA VERİYO
