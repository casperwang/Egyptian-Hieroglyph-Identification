print("Hello")

import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
print("Successful")

#init the CNN
classifier = Sequential()

#start layering up!
classifier.add(Conv2D(
    32, (3, 3), input_shape = (64, 64, 3), activation = "relu")
)

#pooling
classifier.add(MaxPooling2D(
    pool_size = (2, 2)
))

#flattening
classifier.add(Flatten())

#full connection - 
classifier.add(Dense(activation="relu", units=1024))
classifier.add(Dense(activation="relu", units=1024))
classifier.add(Dense(activation="sigmoid", units=50))

#compile
classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])

#getting images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    "C:/Users/caspe/Documents/資訊專題/EgyptianHieroglyphDataset/ExampleSet7/test_set_2",
    target_size = (64, 64),
    batch_size = 32,
    class_mode = "categorical"
)
print()

test_set = test_datagen.flow_from_directory(
    "C:/Users/caspe/Documents/資訊專題/EgyptianHieroglyphDataset/ExampleSet7/train_set_2",
    target_size = (64, 64),
    batch_size = 32,
    class_mode = "categorical"
)

classifier

from IPython.display import display
from PIL import Image
print("HEY")
train_history = classifier.fit_generator(
    training_set,
    steps_per_epoch = 500,
    epochs = 6,
    validation_data = test_set,
    validation_steps = 2000//32
)

print("Done")

import matplotlib.pyplot as plt
def show_train_history(train_history, training_set, test_set) :
    plt.plot(train_history.history[training_set])
    plt.plot(train_history.history[test_set])
    plt.title('Train History')
    plt.ylabel(training_set)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show();
