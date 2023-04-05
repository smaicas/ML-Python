# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist   

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 


train_images = train_images / 255.0    
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'), # 128 neuronas/nodos/variables La función ReLU transforma los valores introducidos anulando los valores negativos y dejando los positivos tal y como entran.
    keras.layers.Dense(10, activation='softmax') #Cada nodo contiene una calificacion que indica la probabilidad que la actual imagen pertenece a una de las 10 clases.
])

model.compile(optimizer='adam', #lo que un optimizador hace es, obviamente, optimizar los valores de los parámetros para reducir el error cometido por la red. El proceso mediante el cual se hace esto se conoce como “backpropagation”La optimización de Adam es un método de descenso de gradiente estocástico que se basa en la estimación adaptativa de momentos de primer y segundo orden.
              loss='sparse_categorical_crossentropy', #Esto mide que tan exacto es el modelo durante el entrenamiento. Quiere minimizar esta funcion para dirigir el modelo en la direccion adecuada.
              metrics=['accuracy']) #Se usan para monitorear los pasos de entrenamiento y de pruebas. El siguiente ejemplo usa accuracy (exactitud) , la fraccion de la imagenes que son correctamente clasificadas.

model.fit(train_images, train_labels, epochs=10)