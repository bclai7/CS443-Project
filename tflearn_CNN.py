import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()


TRAIN_DIR = 'Traffic Sign Data Set/TRAIN-LARGE/'
TEST_DIR = 'Traffic Sign Data Set/TEST/'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'trafficSigns-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match


def label_img(img):
    word_label = img.split('.')[-2]
    # conversion to one-hot array [cat,dog]
    if 'Do_Not_Enter_Sign' in word_label:
        return [1, 0, 0, 0, 0, 0]
    elif 'No_Left_Turn_Sign' in word_label:
        return [0, 1, 0, 0, 0, 0]
    elif 'One_Way_Sign' in word_label:
        return [0, 0, 1, 0, 0, 0]
    elif 'Speed_Limit_25MPH' in word_label:
        return [0, 0, 0, 1, 0, 0]
    elif 'Stop_Sign' in word_label:
        return [0, 0, 0, 0, 1, 0]
    elif 'Yield_Sign' in word_label:
        return [0, 0, 0, 0, 0, 1]
    else:
        return [0, 0, 0, 0, 0, 0]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        if img is not None:
            label = label_img(img)
            if label == [0, 0, 0, 0, 0, 0]:
                continue
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        else:
            print("image data null")
            quit()
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


train_data = create_train_data()
# If you have already created the dataset:
# train_data = np.load('train_data.npy')

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

### EXTRA LAYERS

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

###

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# model.save(MODEL_NAME)

# TESTING
test_data = process_test_data()
fig = plt.figure()

for num, data in enumerate(test_data[:12]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        str_label = 'Do Not Enter'
    elif np.argmax(model_out) == 1:
        str_label = 'No Left Turn'
    elif np.argmax(model_out) == 2:
        str_label = 'One Way'
    elif np.argmax(model_out) == 3:
        str_label = 'Speed Limit 25'
    elif np.argmax(model_out) == 4:
        str_label = 'Stop'
    elif np.argmax(model_out) == 5:
        str_label = 'Yield'
    else:
        str_label = 'Other'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

















