import os
import numpy as np

from skimage import io, transform
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D
from keras.layers import MaxPooling2D, Flatten, Dropout

NB_EPOCH = 10
IMAGE_SIZE = 50
DATA_DIR = "data"
TRAIN_DATA_FRACTION = 0.8


def test_train_split(data, labels, f):
    test_data_size = int(len(data) * f)
    return data[:test_data_size], labels[:test_data_size], \
        data[test_data_size:], labels[test_data_size:]


def CNN():
    model = Sequential()
    model.add(Convolution2D(8, 3, 3, border_mode='same',
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def transform_img(image):
    return transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE, image.shape[2]))


def loadData():
    images = os.listdir(DATA_DIR)
    train_data = []
    train_labels = []
    for image in images:
        if image[-4:] == 'jpeg':
            transformed_image = transform_img(
                io.imread(DATA_DIR + '/' + image))
            train_data.append(transformed_image)
            label_file = image[:-5] + '.txt'
            with open(DATA_DIR + '/' + label_file) as f:
                content = f.readlines()
                label = int(float(content[0]))
                l = [0, 0]
                l[label] = 1
                train_labels.append(l)
    return np.array(train_data), np.array(train_labels)


data, labels = loadData()
train_data, train_labels, test_data, test_labels = test_train_split(
    data, labels, TRAIN_DATA_FRACTION)

print "Train data size: ", len(train_data)
print "Test data size: ", len(test_data)

idx = np.random.permutation(train_data.shape[0])
model = CNN()
model.fit(train_data[idx], train_labels[idx], nb_epoch=NB_EPOCH)

preds = np.argmax(model.predict(test_data), axis=1)
test_labels = np.argmax(test_labels, axis=1)
print accuracy_score(test_labels, preds)
