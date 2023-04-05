# TensorFlow, tf.keras, numpy, pandas, matplotlib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re as re


# Read Full Data from CSV
df=pd.read_csv("C:\Workspace\ML\data-train.csv")
# print(df)

# Split Data Frames: Train, Validation, Test
n = len(df)
train_df = df[0:int(n*0.7)] #70%
val_df = df[int(n*0.7):int(n*0.9)] #20%
test_df = df[int(n*0.9):] #10%

# Standarization
def clean_text(text):
    cleaned_text = re.sub('[^a-zA-Z0-9\s]+', '', text) #Elimina todos los caracteres que no sean letras, números o espacios
    return cleaned_text

train_df['InputText'] = train_df['InputText'].apply(clean_text)
val_df['InputText'] = val_df['InputText'].apply(clean_text)
test_df['InputText'] = test_df['InputText'].apply(clean_text)

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['InputText'])

train_sequences = tokenizer.texts_to_sequences(train_df['InputText'])
val_sequences = tokenizer.texts_to_sequences(val_df['InputText'])
test_sequences = tokenizer.texts_to_sequences(test_df['InputText'])

maxlen = 100
train_padded = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
val_padded = keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=maxlen, padding='post', truncating='post')
test_padded = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=maxlen, padding='post', truncating='post')

model = keras.Sequential([
keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=maxlen),
keras.layers.LSTM(16),
keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_padded, train_df['Label'], epochs=10, validation_data=(val_padded, val_df['Label']))

# Evaluate
test_loss, test_acc = model.evaluate(test_padded, test_df['Label'])
print('Test accuracy:', test_acc)

# Plot Accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss function
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model.predict(test_padded)

# Predict
y_true = test_df['Label']
y_pred = (predictions > 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)