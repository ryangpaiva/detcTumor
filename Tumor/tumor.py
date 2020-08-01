import os
import pandas as pd
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import keras

from google.colab import drive
drive.mount('/content/drive')

caminho ='./drive/My Drive/tumores/'
os.listdir(caminho)

def ftumores(caminho):
 
  subpasta = ['no', 'yes']
 
  imagem = []
  rotulo = []
 
  for sub in subpasta:
    arquivos_img = os.listdir(caminho+sub)
    for imagens in arquivos_img:
      img_array = cv2.imread(caminho+sub+'/'+imagens, cv2.IMREAD_GRAYSCALE)
      img_array = cv2.resize(img_array, (30,20), interpolation = cv2.INTER_AREA)
      img_array = tf.keras.utils.normalize(img_array, axis = 1)
      img_array = img_array.reshape(-1)
      imagem.append(img_array)
      if sub == 'yes':
        rotulo.append(1)
      else:
        rotulo.append(0)
 
  return np.asarray(rotulo), np.asarray(imagem)

rotulo, caracteristicas = ftumores(caminho)

caracteristicas_treino, caracteristicas_teste, rotulo_treino, rotulo_test = train_test_split(caracteristicas, rotulo, test_size = 0.2, random_state = 10)

model = keras.Sequential()
 
model.add(keras.layers.core.Dense(300, input_shape =tuple([caracteristicas.shape[1]]), activation='sigmoid'))
 
model.add(keras.layers.core.Dense(40, activation='sigmoid'))
model.add(keras.layers.core.Dense(12, activation='sigmoid'))
 
model.add(keras.layers.Dropout(rate=0.2))
 
 
model.add(keras.layers.core.Dense(2, activation='sigmoid'))
 
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

model.fit(caracteristicas_treino, rotulo_treino, epochs=90)

model.evaluate(caracteristicas_treino, rotulo_treino)

predictions = model.predict(caracteristicas_teste)
predictions[0:5]

rotulo_pred = ["livre de tumor" if x[0] > x[1] else "com tumor" for x in predictions]
rotulo_pred[0:10]

classifications = model.predict(caracteristicas_teste)
print(classifications[50])

# agora vamos pegar uma imagem real, que já sabemos o resultado e vamos testar se a rede neural reconhece de forma correta

from urllib.request import urlretrieve
 
urlretrieve("https://drive.google.com/u/0/uc?id=1DTZ6X7Tunu0GdnxYD7MUE0NI1b_bVsJv&export=download", "tumor.jpg")

tumor = cv2.imread('tumor.jpg',cv2.IMREAD_GRAYSCALE )
print(tumor)

%matplotlib inline
plt.imshow(tumor)
plt.show()

tumor = cv2.resize(tumor, (30,20), interpolation = cv2.INTER_AREA) 
len(tumor)

tumor = tf.keras.utils.normalize(tumor, axis=1) 
tumor

tumor= tumor.reshape(-1)
tumor

caracteristicas_teste = np.vstack((caracteristicas_teste, tumor))

len (classifications)

print(classifications[50]) # aqui vamos ver uma predição certa

# agora podemos fazer outro teste com um resultado negativo

from urllib.request import urlretrieve

urlretrieve("https://drive.google.com/u/0/uc?id=1zfOFjePn9CfFTAeQq48Rmxj7CfdGJoc4&export=download", "nada.jpeg")

nada = cv2.imread('nada.jpeg',cv2.IMREAD_GRAYSCALE )
print(tumor)

%matplotlib inline
plt.imshow(nada)
plt.show()

nada = cv2.resize(nada, (30,20), interpolation = cv2.INTER_AREA) 
len(nada)

nada = tf.keras.utils.normalize(nada, axis=1) 
nada

nada= nada.reshape(-1)
nada

caracteristicas_teste = np.vstack((caracteristicas_teste, nada))

print(classifications[52]) # aqui teremos uma previsão que indica negativo


