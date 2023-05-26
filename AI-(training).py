import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pandas as pd
from keras.optimizers import SGD

#loads stored X data for usage
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

#loads stored y data
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0


#trains the model through 6 layers - optimized do not change
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

#converts data to array to bypass error - do not touch
X = np.array(X)
y = np.array(y)

#model is fitted with optimized variables
model_history = model.fit(X, y, batch_size=32, epochs=100, validation_split=0.3)

#model is saved for further usage
model.save('6xn-CNN.model')

#model shows all data for developer analysis
print(model.summary()) 
pd.DataFrame(model_history.history).plot(figsize=(9,6))
plt.show()
