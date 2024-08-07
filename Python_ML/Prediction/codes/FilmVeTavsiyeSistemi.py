import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/users.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
filmler = pd.read_csv("/Users/enesbal/Desktop/m_learn/csv_files/movie_id_titles.csv")

data = pd.merge(data, filmler, on="item_id")

tablo = data.pivot_table(index="user_id", columns="title", values="rating")

starwars = tablo["Star Wars (1977)"]
korelasyon = tablo.corrwith(starwars)

kore = pd.DataFrame(korelasyon, columns=["Correlation"])
kore.dropna(inplace=True)

data = data.drop(["timestamp"], axis=1)

ratings = pd.DataFrame(data.groupby("title")["rating"].mean())
ratings.sort_values("rating", ascending=False).head(15)

ratings["oysayısı"] = pd.DataFrame(data.groupby("title")["rating"].count())
ratings.sort_values("oysayısı", ascending=False).head(15)
print(ratings)

kore = kore.join(ratings["oysayısı"])
kore[kore["oysayısı"] > 100].sort_values("Correlation", ascending=False).head()
print(kore)
