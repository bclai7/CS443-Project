import tensorflow as tf
import cv2
import numpy as np
import os
from random import shuffle
import re
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

tf.reset_default_graph()

train_location = 'Traffic Sign Data Set/TRAIN/'
test_location = 'Traffic Sign Data Set/TEST/'

x_pixels = 50
y_pixels = 50

learn_rate = 1e-3

model_name = 'trafficsigns-{}-{}.model'.format(learn_rate, '2conv-basic')

def label_img(img):
    word_label = img.split('_')[0]
    # conversion to one-hot array [Do Not Enter, No Left Turn, One Way, Speed Limit, Stop, Yield]
    if word_label == 'Do':
        return [1, 0, 0, 0, 0, 0]
    elif word_label == 'No':
        return [0, 1, 0, 0, 0, 0]
    elif word_label == 'One':
        return [0, 0, 1, 0, 0, 0]
    elif word_label == 'Speed':
        return [0, 0, 0, 1, 0, 0]
    elif word_label == 'Stop':
        return [0, 0, 0, 0, 1, 0]
    elif word_label == 'Yield':
        return [0, 0, 0, 0, 0, 1]

def create_train_data():
    training_data = []
    for img in os.listdir(train_location):
        label = label_img(img)
        path = os.path.join(train_location,img)
        img = cv2.imread(path)
        img = cv2.resize(img, (x_pixels, y_pixels))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in os.listdir(test_location):
        path = os.path.join(test_location, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path)
        img = cv2.resize(img, (x_pixels, y_pixels))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data=create_train_data()
test_data=process_test_data()

convnet = input_data(shape=[None, x_pixels, y_pixels, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=learn_rate, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


#############
'''
if os.path.exists('{}.meta'.format(model_name)):
    model.load(model_name)
    print('model loaded')
'''

train = train_data
test=test_data

#print(len(train))
#print(len(test))

# for i in train:
#     print('%s' % i)
#     print(i)
#     print(len(i))

X = np.array([i[0] for i in train]).reshape(-1,x_pixels,y_pixels,1)
Y = [i[1] for i in train]

print('len X')
print(len(X))

test_x = np.array([i[0] for i in test]).reshape(-1,x_pixels,y_pixels,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=model_name)
print ('HERE')
model.save(model_name)













