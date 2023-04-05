# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# to do --> arreglar el bug, meterle el validation al train, hacerle plot(graficos) al fit para mirar el .history (las metricas), aumentar el set de datos.

df=pd.read_csv("C:\Workspace\ML\data-train.csv")
# print(df)

n = len(df)
train_df = df[0:int(n*0.7)] #70%
val_df = df[int(n*0.7):int(n*0.9)] #20%
test_df = df[int(n*0.9):] #10%

# Featurize text
max_features = 10000
sequence_length = 250
train_text = train_df["InputText"].values

vectorize_layer = tf.keras.layers.TextVectorization(output_mode="int", output_sequence_length=100, standardize=None)
# Convierte en un conjunto de datos de TensorFlow
# text_ds = tf.data.Dataset.from_tensor_slices(train_df["InputText"]) #no hace falta que hagas esto haces, df["werwer..."].values y sacas los textos asi lo ves mejor
vectorize_layer.adapt(train_text)
#solo te muestro como sacarlo a fuera lo puedes hacer todo asi o meterlo como capa en el squential
vectorized = vectorize_layer(train_text)
normalization_layer=tf.keras.layers.Normalization()
normalization_layer.adapt(vectorized) # dale a ver pero creo q esto no es asi
print(vectorized[0])

# for vector in vectorized:
    # print(vector)


normalized = normalization_layer(vectorized)
print(normalized[0]) 

# for norm in normalized:
    # print(norm) 

train_text=train_df["InputText"].values
train_labels=train_df["Label"].values
val_texts=val_df["InputText"].values
val_labels=val_df["Label"].values

# Crea modelo
modelotest = tf.keras.models.Sequential([
    vectorize_layer,
    normalization_layer,                       # Añade capa de normalización de lote después de la capa de vectorización de texto
    tf.keras.layers.Embedding(max_features+1,output_dim=16), 
    tf.keras.layers.GlobalAveragePooling1D(), 
    tf.keras.layers.Dense(1, activation="sigmoid")
])
modelotest.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

EPOCHS=10
history=modelotest.fit(train_text, train_labels, epochs=EPOCHS, validation_data=(val_texts,val_labels), shuffle=True)
history.history # aqui sacas los datos haces un diccionario y lo mandas al plot 
###############raura######################
###############raura######################


# train_text = train_df["InputText"].values
# vectorize_layer.adapt(train_text)

# print(vectorize_layer(train_text))


# me falta la normalizacion pero esto deberia ir


###############raura######################
###############raura######################


# print(tf.Tensor(vectors[0]))
# for data in train_text:


# def vectorize_text(text, label):
#   text = tf.expand_dims(text, -1)
#   return vectorize_layer(text), label

# # retrieve a batch (of 32 reviews and labels) from the dataset
# text_batch, label_batch = next(iter(train_df[0:len(train_df)]))
# first_text, first_label = text_batch[0], label_batch[0]
# print("Input", first_text)
# print("Label", train_df.class_names[first_label])
# print("Vectorized review", vectorize_text(first_text, first_label))

# # JS, Py, Java, C#
# ejemplo = tf.keras.layers.Hashing(num_bins=(4))


# vectorize_layer = tf.keras.layers.TextVectorization(
#     # standardize=custom_standardization,
#     max_tokens=max_features,
#     output_mode='int',
#     output_sequence_length=sequence_length)

# train_mean = train_df.mean() #media
# train_std = train_df.std() #varianza
# train_df=(train_df - train_mean) / train_std
# val_df=(val_df - train_mean) / train_std
# test_df=(test_df - train_mean) / train_std
