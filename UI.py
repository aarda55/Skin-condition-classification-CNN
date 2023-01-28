import cv2
import tensorflow as tf
import PySimpleGUI as sg
import numpy as np
import os
import io
from PIL import Image


#loads up model
model = tf.keras.models.load_model("6xn-CNN.model(80% accuracy)")
CATEGORIES = ["acne and Rosacea", "skincancer","Eczema","Psoriasis","Actinic Keratosis","Nail Fungus","Tinea Ringworm","Seborrheic Keratoses"]
file_types = [
    ("JPEG (*.jpg)", "*.jpg"),
    ("All files (*.*)", "*.*")
]


#defines simple UI for test porpuses
layout = [
    [sg.Image(key="-IMAGE-")],
    [   
        sg.Text("Image File"),
        sg.Input(size=(25, 1), key="-FILE-"),
        sg.FileBrowse(file_types=file_types),
        sg.Button("Detect"),    
    ],
]

window = sg.Window("Derma-AI(DAI)", layout)

def prepare(filepath):
    IMG_SIZE = 80
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "Detect":
        filename = values["-FILE-"]
        if os.path.exists(filename):
            image = Image.open(values["-FILE-"])
            bio = io.BytesIO()
            # Actually store the image in memory in binary 
            image.save(bio, format="PNG")
            prediction = model.predict([prepare(values["-FILE-"])])
            print(CATEGORIES[int(prediction[0][0])])

'''def prepare(filepath):
    IMG_SIZE = 80
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
prediction = model.predict([prepare(values["-FILE-"])])
print(CATEGORIES[int(prediction[0][0])])'''


window.close()