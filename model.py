#importing libraries and frameworks

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import * #import everything from keras.optimizers
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D

from scipy.misc import imread, imsave
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random

#################################################################

def flipped(image, measurement):
  return np.fliplr(image), -measurement

def get_image(i, data):

  positions, corrections = ['left', 'center', 'right'], [.25, 0, -.25]
  ID, r = data.index[i], random.choice([0, 1, 2])

  measurement = data['steering'][ID] + corrections[r]

  path = PATH + data[positions[r]][ID][1:]
  if r == 1: path = PATH + data[positions[r]][ID]
  image = imread(path)

  if random.random() > 0.5:
    image, measurement = flipped(image, measurement)

  return image, measurement

#################################################################

def generate_samples(data, batch_size):

  while True:

    SIZE = len(data)
    data.sample(frac = 1)

    for start in range(0, SIZE, batch_size):
      images, measurements = [], []

      for this_id in range(start, start + batch_size):
        if this_id < SIZE:
          image, measurement = get_image(this_id, data)
          measurements.append(measurement)
          images.append(image)

      yield np.array(images), np.array(measurements)

#################################################################
# Create the Sequential (NN) model
model = Sequential()

# Adding layers to the model using add() function
# Cropping the image - output_shape = (65, 320, 3)
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
# Normalize - output_shape = (65, 320, 3)
model.add(Lambda(lambda x: (x / 127.5) - 1.))
# 2D convolution layer - output_shape = (17, 80, 16)
model.add(Convolution2D(16, 8, 8, subsample = (4, 4), border_mode = "same"))
# Activation layer (Exponential Linear Units) - output_shape = (17, 80, 16)
model.add(ELU())
# 2D convolution layer - output_shape = (9, 40, 32)
model.add(Convolution2D(32, 5, 5, subsample = (2, 2), border_mode = "same"))
# Activation layer (Exponential Linear Units) - output_shape = (9, 40, 32)
model.add(ELU())
# 2D convolution layer - output_shape = (5, 20, 64)
model.add(Convolution2D(64, 5, 5, subsample = (2, 2), border_mode = "same"))
# Flattening the input - output_shape = 6400
model.add(Flatten())
# Dropout the input at 0.2 rate - output_shape = 6400
model.add(Dropout(.2))
# Activation layer (Exponential Linear Units) - output_shape = 6400
model.add(ELU())
# Fully connected layer - output_shape = 512
model.add(Dense(512))
# Dropout the input at 0.5 rate - output_shape = 512
model.add(Dropout(.5))
# Activation layer (Exponential Linear Units) - output_shape = 512
model.add(ELU())
# Fully connected layer - output_shape = 1
model.add(Dense(1))

model.summary()
model.compile(optimizer = "adam", loss = "mse")

#################################################################

BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 10
PATH = "./"
CSV_FILE = "driving_log.csv"

DATA = pd.read_csv(PATH + CSV_FILE, usecols = [0, 1, 2, 3])

training_data, validation_data = train_test_split(DATA, test_size = 0.15)
total_train = len(training_data)
total_valid = len(validation_data)

#################################################################
print('Training model...')

training_generator = generate_samples(training_data, batch_size = BATCH_SIZE)
validation_generator = generate_samples(validation_data, batch_size = BATCH_SIZE)

history_object = model.fit_generator(training_generator,
                 samples_per_epoch = total_train,
                 validation_data = validation_generator,
                 nb_val_samples = total_valid,
                 nb_epoch = NUMBER_OF_EPOCHS,
                 verbose = 1)

#################################################################
print('Saving model...')

model.save("model.h5")

with open("model.json", "w") as json_file:
  json_file.write(model.to_json())

print("Model Saved.")