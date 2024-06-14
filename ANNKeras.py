# author : ertugrulkirac

#https://www.kaggle.com/code/sumitkant/simple-neural-network-with-keras-and-tensorflow/input

#Building a simple NN model using Keras & TensorFlow
#TensorFlow ve Keras kullanarak adım adım basit bir yapay sinir ağı modeli oluşturma işlemi

import pandas as pd
import numpy as np

#Adım 1: Veri kümesini yükleme ve eğitim ve doğrulama kümelerine bölebilme

# Veri kümesini yükleme
df = pd.read_csv('diabetes.csv')
print(df.head())

print ('Number of Rows :', df.shape[0])
print ('Number of Columns :', df.shape[1])
print ('Number of Patients with outcome 1 :', df.Outcome.sum())

print(df.describe())

"""
count: Boş olmayan gözlemlerin sayısı.
mean: Değerlerin ortalaması.
std: Gözlemlerin standart sapması.
max: Nesnedeki değerlerin maksimumu.
min: Nesnedeki değerlerin minimumu.
"""
# Veri kümesini eğitim ve doğrulama kümelerine bölme
from sklearn.model_selection import train_test_split
X = df.to_numpy()[:,0:8]  # input olarak ilk 8 sütun input olarak kullanılıyor.
Y = df.to_numpy()[:,8]  # output olarak son sütun kullanılıyor.

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, random_state = 0)
print (f'Shape of Train Data : {X_train.shape}')
print (f'Shape of Test Data : {X_test.shape}')

# Adım 2: Keras ile modelin tanımlanması
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

# Sıralı bir model oluşturma
model = Sequential([
    Dense(24, input_dim = (8), activation = 'relu'), # Giriş katmanı : 8 girdi var.
    Dense(10, activation = 'relu'),
    Dense(1, activation = 'sigmoid'), # Çıkış katmanı : 1 çıktı var.
])

# sigmoid activation function is used at the last dense because this is a classification problem.
# sınıflandırma problemi olduğu için son katmanda sigmoid aktivasyon fonksiyonu kullanıldı.

epoch_n =500

#Adım 3: Modeli derleme

# Modeli derleme

model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

#Adım 4: Modeli eğitme

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=epoch_n, batch_size=12, verbose = 2)


# Adım 5: Modeli doğrulama veri kümesiyle değerlendirme
scores = model.evaluate(X_test, y_test)

print(model.metrics_names) #--> ['loss', 'compile_metrics'] there are two types of metrics_name
                           # iki farklı metrics türü bulunuyor.

# Model skor metriğini seçme ve skorunu hesaplama.
print (f'{model.metrics_names[1]} : {round(scores[1]*100, 2)} %') # model.metrics_names[1] = 'compile_metrics'
                                                                  # choosen metrics_names

# Bu adımları kullanarak, TensorFlow ve Keras ile basit bir yapay sinir ağı modeli oluşturabilir
# ve eğitebilirsiniz.


import matplotlib.pyplot as plt

# Plotting loss
plt.plot(history.history['loss'])
plt.title('Binary Cross Entropy Loss on Train dataset')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# Plotting accuracy metric
plt.plot(history.history['accuracy'])
plt.title('Accuracy on the train dataset')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()