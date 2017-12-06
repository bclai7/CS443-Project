import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import cv2

# dimensions of our images.
img_width, img_height = 50, 50

train_data_dir = 'Traffic Sign Data Set/ALL SIGNS-TRAIN/'
validation_data_dir = 'Traffic Sign Data Set/ALL SIGNS-TEST/'

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit_generator(
        train_generator,
        epochs=5,
        validation_steps=5,
        steps_per_epoch=3)

model.save_weights('models/basic_cnn_20_epochs.h5')
# model.load_weights('models_trained/augmented_30_epochs.h5')
model.evaluate_generator(validation_generator, 55)

img = image.load_img('Traffic Sign Data Set/ALL SIGNS-TEST/Test/5.jpg')
img = np.reshape(img,[1,50,50,3])
prediction=model.predict(img)
print(prediction)