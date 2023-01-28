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
import time

DATADIR = "Datasets"
CATEGORIES = ["acne and Rosacea", "skincancer","Eczema","Psoriasis","Actinic Keratosis","Nail Fungus","Tinea Ringworm","Seborrheic Keratoses"]
IMG_SIZE = 80

training_data = []

def create_training_data():
   for category in CATEGORIES:
      path = os.path.join(DATADIR, category)
      class_num = CATEGORIES.index(category)
      for img in os.listdir(path):
          try:
              img_array = cv2.imread(os.path.join(path,img))
              new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
              training_data.append([new_array, class_num])
          except Exception as e:
              pass

create_training_data()
random.shuffle(training_data)
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()