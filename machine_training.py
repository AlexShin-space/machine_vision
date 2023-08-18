import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

#collect our data
#collect nums from mnist
(X_nums_train, y_nums_train), (X_nums_test, y_nums_test) = mnist.load_data()
#collect letters from local csv dataset
X_letters_train = []
y_letters_train = []
for row in open('path to downloaded "A_Z Handwritten Data.csv" file'): #https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
  try:
    row = row.split(",")
    label = int(row[0])
    image = np.array([int(x) for x in row[1:]], dtype="uint8")
    image = image.reshape((28, 28))
    X_letters_train.append(image)
    y_letters_train.append(label)
  except:
    print("ex")
X_letters_train = np.array(X_letters_train, dtype='float32')
y_letters_train = np.array(y_letters_train, dtype="int")


#joining our datasets
y_nums_train += 26 #there are 26 letters in english alphabet
data = np.vstack([X_letters_train, X_nums_train])
labels = np.hstack([y_letters_train, y_nums_train])
data = np.array(data, dtype="float32")
data /= 255.0 #to change matrix in [0:1] range

#to generate changed images from dataset
aug = ImageDataGenerator(
rotation_range=10,
zoom_range=0.05,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.15,
horizontal_flip=False,
fill_mode="nearest")

batch_size=128
vector_labels = keras.utils.to_categorical(labels, 36)
expand_data = np.expand_dims(data, -1)

#architecture of neural network
model = keras.Sequential(
    [
        keras.Input(shape=np.expand_dims(data[0], -1).shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dense(256, activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(36, activation="softmax"),
    ]
)
print(model.summary())
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

#train our model
model.fit(
aug.flow(expand_data, vector_labels, batch_size=batch_size),
validation_data=(expand_data, vector_labels),
steps_per_epoch=len(expand_data)/batch_size, epochs=5,
verbose=1, validation_split=0.1)
