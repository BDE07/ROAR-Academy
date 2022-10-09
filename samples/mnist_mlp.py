## This is course material for Introduction to Modern Artificial Intelligence
## Example code: mnist_mlp.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 40

# input image dimensions
img_rows, img_cols = 28, 28

# load the data built in Keras, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display some examples from first 20 images
print(y_train[0:20])
plt.figure(1)
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(x_train[i], cmap = plt.cm.binary)
plt.show()

x_train = x_train.reshape(x_train.shape[0], img_rows*img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows*img_cols)
input_shape = (img_rows*img_cols,)

# When calculating image data, convert from uint8 to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Reduce the element range from [0, 255] to [0, 1]
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(800, input_shape=input_shape, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax' ))

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)

print(model.summary())
print('Test loss:', score[0])
print('Test accuracy:', score[1])
