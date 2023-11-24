import tensorflow as tf
import keras 
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

import os.path as path

class_names = ['T-shirt', 'pantalon', 'pull', 'robe', 'manteau', 'sandale', 'chemise', 'basket', 'sac', 'bottine']

mnist = keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images  = training_images / 255.0
test_images = test_images / 255.0


if (path.isfile("fashion_model.keras")):
  print("hello")
  model = keras.models.load_model("fashion_model.keras")
else :
  model = keras.models.Sequential([
    layers.Flatten(), 
    layers.Dense(256, activation="relu"), 
    layers.Dense(10, activation="softmax")
  ])

model.compile(
  optimizer = "adam",
  loss = 'sparse_categorical_crossentropy',
  metrics=['accuracy']
)

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

model.save("fashion_model.keras")

classifications = model.predict(test_images)

t= 2
print(classifications[t])
print("L'image correspond à " + str(100*np.max(classifications[t])) + "% à un(e) " + class_names[test_labels[t]])

plt.figure()
plt.imshow(test_images[t])
plt.colorbar()
plt.grid(False)
plt.show()