import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import tensorflow as tf
from tensorflow.keras.models import Sequential
# Model oluşturuyoruz
from tensorflow.keras.layers import Dense
# Modelin içine katmananları koyuyoruz
from tensorflow.keras.models import load_model

# DataFrame'i oluştur
dataFrame = pd.read_excel("/Users/enesbal/Desktop/Deep/Excel_Files/bisiklet_fiyatlari.xlsx")

# İlk beş satırı yazdır
print(dataFrame.head())

# Çift değişkenli ilişkileri gösteren pairplot'u çiz
sbn.pairplot(dataFrame)
# pairplot : çift çizim işlemiyle grafik gösterir

# Grafiği görüntüle
#plt.show() # pc kasmasın diye şimdilik kapadım açabilirsin tekrardan .d

from sklearn.model_selection import train_test_split
# Veriyi test ve training setine bölme
# Formül : y = wx + b
x = dataFrame[["BisikletOzellik1","BisikletOzellik2"]].values
# x: özellik ("feature")
# values demezsek pandas series olur, values yazarsak numpy dizisi olur
y = dataFrame["Fiyat"].values
# y gitmek istediğimiz nokta ("label")
# y: hedef ("target variable")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=15)
# test_size  parametresi ile %33'ün ne kadar verinin test ayrılmaya alınacağını belirtiyoruz.
# random_state değeri vermeye gerek bile yok (verilerin seçimiyle ilgili ama önemsiz/grafik görünümü biraz değişir)
# random_state isteğe bağlı parametre sayesinde her zaman aynı sonucu vermeyebilir.Genelde 42 yazılır

print(x_train.shape)
# x_train için kullanılan veri sayısı gözükür
print(x_test.shape)
# x_test için kullanılan veri sayısı gözükür
print(y_train.shape)
# y_train için kullanılan veri sayısı gözükür
print(y_test.shape)
# y_test için kullanılan veri sayısı gözükür
# Eğer "x" için shape demezsek matrix gösterir, çünkü özellik 1 ve özellik 2 olmak üzere 2 değer var

# Scalling işlemi yapmamız lazım (boyutunu değiştirmek)
# veriler, kendi boyutları oranında 0 ve 1 arasında bir rakama yuvarlanıyor 
# Böylece daha doğru bilgileri getirebiliyor
# Bu durumda normalize ederek bu sıklıkla bulunmayacak şekilde yardımcı olur (ai)

from sklearn.preprocessing import  MinMaxScaler
# process : işlemek / preprocessing : işlemekten önce yapılacak işlemler
scaler=MinMaxScaler()
scaler.fit(x_train)
# scaler'ı fit ederek x_train'e uygun hale getiriyoruz / fit : ayarlamak

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
# transform : veri tipi dönüşümü yapıyor (glb)

# Model oluşturuyoruz
model = Sequential()
# model : modele sahip class
# Sequential : modeli elde etmek için kullandığımız class
# Kendini programlayabilen bir sınıf

# Birinci katman ekliyoruz (Ai)
# input_dim : girişimiz kaç tane özellik olduğuna dair bilgi veriyor
#model.add(Dense(6,input_dim=8,kernel_initializer='normal',activation='relu'))
# 6: Katmandaki nöron sayısı. Bu, katmanın çıkış boyutunu belirler.
# input_dim=8: Giriş özelliklerinin sayısı. Bu durumda, 8 adet giriş özelliği olduğunu belirtiyoruz. Bu parametreyi vermeniz tavsiye edilir (ai).
# kernel_initializer='normal': Ağırlıkların başlangıç değerlerini belirleyen strateji. 'normal', normal bir dağılıma sahip rastgele değerlerle başlamak anlamına gelir.
# activation='relu': Aktivasyon fonksiyonu olarak ReLU (Rectified Linear Unit) fonksiyonunu kullanıyoruz. Bu, katmanın çıkışını oluşturan aktivasyon fonksiyonunu belirler.

model.add(Dense(4,activation='relu')) # Katman-1
model.add(Dense(4,activation='relu')) # Katman-2
model.add(Dense(4,activation='relu')) # Katman-3

model.add(Dense(1)) # Çıkış Katman

model.compile(optimizer="rmsprop",loss="mean_squared_error") #loss="mse" de aynı şey
# Yapılan işemleri birleştirip çalışmaya hazır hale getirir
# optimizer : neye göre optimize edileceğine dair bilgi verir
# rmsprop : hızlıca optimize eder ama sonra da biraz daha yavaşlatır
# MSE Formül :  ((Y - Y')^2)/N (tam olmadı ama bu)

print("Model Oluşturuldu.")

# Verileri eğitimi başlatıyoruz
#history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=1)
# validation_data : eğitime geçirdiğimiz zaman hangi verilere baktığını belirtir
# epochs : kaç adım atacağız?
# batch_size : her defada kaç tane örnek üzerinde işlem yapacaksınız?

model.fit(x_train,y_train,epochs=250)
#print(model.history.history)
#loss değerlerini dictionary (sözlük) içinde gösterir
loss = model.history.history["loss"] # dizi haline getirip loss değerine atadık

sbn.lineplot(x=range(len(loss)),y=loss)
# eğitilmiş verilerin (x_train,y_train) grafiği
# Eğitimin bitmesinden ve modelin doğru çalışmasından emin olmak için kullanabiliriz
plt.show()
# plt.savefig('LossGraph.png') # grafiği png olarak kaydeder

#model.evaluate(x_train,y_train) # verbose değeriini girmezsek default 1 olur

# Modelin train/test verileri

trainLoss = model.evaluate(x_train,y_train,verbose=0)
# loss değeri verip bir rakam döndürüyor eğer "verbose=0" yazmassan
# yazarsan sadece değer döndürür, bizim için önemli olan da o
testLoss = model.evaluate(x_test,y_test,verbose=0)

print(f"trainLoss : {trainLoss}")
print(f"trainLoss : {testLoss}")
# verilerin birbirine yakın olması önemli

# Verilerin doğruluğunu ispatlama

testTahminleri=model.predict(x_test)
# x_test i veriyoruz, özelliklerinden y_test'i çıkarmaya çalışıyoruz
# predict : tahmin etmek
# testTahminleri : yaptığımız tahminler

print(testTahminleri)

tahminDF = pd.DataFrame(y_test,columns=['Gerçek y']) 
# y_test 'yi DataFrame formatına çevirerek yeni bir dataframe oluştur
print(tahminDF)

testTahminleri=pd.Series(testTahminleri.reshape(330,))

tahminDF=pd.concat([tahminDF,testTahminleri],axis=1)

tahminDF.columns=['Tahmin y','Gerçek y']
print(f"tahminDF : {tahminDF}")

sbn.scatterplot(x="Gerçek y",y="Tahmin y",data=tahminDF)
# Grafik çizgi olarak çıktıysa modelimiz genel olarak doğru çalışmaktadır

# Gerçek hata değerini bulma
from sklearn.metrics import mean_absolute_error, mean_squared_error
MAE=mean_absolute_error(tahminDF["Gerçek y"],tahminDF["Tahmin y"])
# gerçek ve tahmin arasındaki farkı alıp ortalar
# MAE (Mean Absolute Error)

MSE=mean_squared_error(tahminDF["Gerçek y"],tahminDF["Tahmin y"])
# gerçek ve tahmin arasındaki kare farkını alıp toplayıp üçüncü tarafın katlanarak ortalar
# MSE (Mean Squared Error)

print(f"MAE : {MAE}")
print(f"MSE : {MSE}")

print(dataFrame.describe())
# ortalama değerleri verir (her türden değeri verir)

#   --> 8.13 - Model Tahminleri <--
yeniBisikletOzellikleri=[[1760,1758]]
# rastgele bisiklet özellik değerleri atıyoruz
yeniBisikletOzellikleri=scaler.transform(yeniBisikletOzellikleri)
# yukarıda eşitlediğimiz sayıları küçültüp kendince bir rakama eşitler

print(model.predict(yeniBisikletOzellikleri))
# atadğımız sayılara göre çıktı veriyor (eğittiğimiz modeli kullanarak)
# Bu sayılara göre ne tür bir bisiklet olduğunu, modelin neye bakacağı, modelin ne tahmin ettiği (ai)

model.save("bisiklet_modeli.h5")
# modelimizi kaydediyoruz (h5 uzantısı ile kaydedilir)

sonradanCagirilanModel=load_model("bisiklet_modeli.h5")
# Kaydedilen modeli tekrar cagrıyoruz

print(sonradanCagirilanModel.predict(yeniBisikletOzellikleri))
# gene aynı özellikleri (değerleri) döndürür




# Modeli test edilmesi (Ai)
# 2 tane (/n) var düzeltip test edebilirsin -> \n yap
"""
# Sonucu test edelim
predictions = model.predict(x_test)
# x_test'teki verileri modelle sınamayıp yani predict et
# predictions : model tarafından verdiği cevaplar

# Cevaplarımızı yükseltmek veya düşürmek için threshold koyuyoruz
threshold = 0.5
predicted_classes = [1 if p[0] > threshold else 0 for p in predictions]
true_classes = y_test

from sklearn import metrics

print("Test Sonuçları: ")
print("Tahmin Edilen Sınıflar : ", predicted_classes)
print("Gerçek Sınıflar        : ", true_classes)

confusion_matrix = metrics.confusion_matrix(true_classes, predicted_classes)
print("(/n)Çoklu Klasifik asyon Matrisi :")
print(confusion_matrix)

# TP FP TN FN
# True Positive False Negative
# Gerçek negatif sayısı gerçek pozitif sayıs ıdır.
TP = confusion_matrix[0][0]    #True positive
FP = confusion_matrix[0][1]   #False positive
FN = confusion_matrix[1][0]    #False negative
TN = confusion_matrix[1][1]    #True negative   

print('''(/n)Katsayılar;
Accuracy     = {0:>4}
Precision    = {1:>4}
Recall       = {2:>4}
F1 Score     = {3:>4}
'''.format(metrics.accuracy_score(true_classes, predicted_classes),
           metrics.precision_score(true_classes, predicted_classes),
           metrics.recall_score(true_classes, predicted_classes),
           metrics.f1_score(true_classes, predicted_classes)))

# ROC Kurve (Receiver Operating Characteristics Curve)
# Görsel olarak doğru ve yanlış tahminler arasındaki performansı gösterir.
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

false_positive_rate = []
true_positive_rate = []

for thresh in np.arange(0, 1 + 0.01, 0.01):
    false_positive_rate.append(1 - specificity(y_test, predicted_probs > thresh))
    true_positive_rate.append(sensitivity(y_test, predicted_probs > thresh))

plt.figure()
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# AUC Değeri (Area Under the Curve)
# Modelin toplam performansını ifade eder.
auc_value = auc(false_positive_rate, true_positive_rate)

print("AUC değeri :", auc_value)
"""






# Bozuk

"""
testTahminleri=model.predict(y_test)
# predict : tahmin etmek
# testTahminleri : yaptığımız tahminler

# Tahminleri skaler formata dönüştürme
skalerTestTahminleri= tf.argmax(testTahminleri,axis=1).numpy().reshape(-1,1)
# reshape (-1,1) : sütun sayısını  1 yapar

# Doğruluk oranını hesaplama
from sklearn.metrics import accuracy_score
doğrulukOranı=accuracy_score(y_true=y_test , y_pred=skalerTestTahminleri)

print("Doğruluk Oranı: ",doğrulukOranı)

# Grafikle gösterme
import matplotlib.pyplot as plt

plt.scatter(y_test,skalerTestTahminleri,color="red")
plt.title("Yanlış Tahmin vs Doğru Tahmin",fontsize=20)
plt.xlabel("Doğru Yatırımlar",fontsize=15)
plt.ylabel("Yanlış Yatırımlar",fontsize=15)
plt.grid(True)
plt.show()

# Eğitime katılmaya devam edelim
# Daha büyük bir miktarda veriye sahibiz ve daha fazla tekrar deneyebilirsiniz

"""

