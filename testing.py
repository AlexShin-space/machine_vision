import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils.image_utils import img_to_array, load_img
import PIL.ImageOps
import os
import sys
from keras.datasets import mnist

tf.keras.utils.disable_interactive_logging() #hide the log
model = keras.models.load_model("model.h5") #import the model

user_input_path = sys.argv[1] #parameter of script running
file_paths = [f.path for f in os.scandir(user_input_path) if not f.is_dir()]

#each index of alldata = index of predicted data
alldata = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' 

#find all images and predict the result of each image
for file_path in file_paths:
  extension = file_path.split('.')[-1]
  if extension == 'jpg' or extension == 'jpeg' or extension == 'png':
    img = load_img(file_path).convert('L')
    img = PIL.ImageOps.invert(img)
    img = img_to_array(img.resize((28,28))) #resize picture to matrix 28*28 pixels
    img = np.squeeze(img, axis=2) #deleting one axis
    img /= 255.0
    predicted = model.predict(np.array([img]))[0]
    symbol_output = alldata[predicted.tolist().index(max(predicted))]
    print(f"{ord(symbol_output)}, {file_path}")