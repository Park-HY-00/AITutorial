import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(7)

""" version check """
# print('Python version : ', sys.version)
# print('TensorFlow version : ', tf.__version__)
# print('Keras version : ', keras.__version__)

img_rows = 28
img_cols = 28

""" Preprocessing 1 : Load the dataset from MNIST """
(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()
# Show the data
plt.imshow(X_train[0], cmap='Greys')
# plt.show()

""" Preprocessing 2 : Process the Image"""
# Show the image by matrix
# for x in X_train[0]:
#     for i in x:
#         sys.stdout.write('%d\t' %i)
#     sys.stdout.write('\n')

""" Preprocessing 3 : Data transformation """
input_shape = (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
# Normalization
X_train = X_train.astype('float64') / 255
X_test = X_test.astype('float64') / 255

print('X_train shape: ', X_train.shape)
print(X_train.shape[0], 'train smaples')
print(X_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 12

""" Preprocessing 4 : Data Encoding by One-Hot Encoding """
y_train = np_utils.to_categorical(Y_class_train, 10)
y_test = np_utils.to_categorical(Y_class_test, 10)

""" Configure the Learning Model :CNN """
model = Sequential()
# model.add(Dense(512, input_dim = 784, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# Add convolution layer
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',  input_shape=input_shape))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Drop out
model.add(Dropout(0.25))
# Flatten
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

""" Learning """
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

""" Check the loss / accuracy """
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

""" Check the prediction """
n = 0
plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
print('The Answer is ', model.predict(X_test[n].reshape((1, 28, 28, 1))))   # 결과값 출력해야함