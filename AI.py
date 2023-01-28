import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
import pandas as pd
from keras.optimizers import SGD

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) 

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('softmax'))

opt = SGD(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

X = np.array(X)
y = np.array(y)
model_history = model.fit(X, y, batch_size=4, epochs=10, validation_split=0.3)

model.save('6xn
-CNN.model')
print(model.summary()) 
pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.show()