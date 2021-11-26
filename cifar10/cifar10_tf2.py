import tensorflow as tf
from tensorflow import keras

from keras.datasets import cifar10

import numpy as np

batch_size = 100
num_classes = 10
epochs = 10

(x_train, train_labels), (x_test, test_labels) = cifar10.load_data()

print(x_train.shape)

train_images = x_train.reshape([-1,32,32,3]) / 255.0
test_images = x_test.reshape([-1,32,32,3]) / 255.0

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(24)

model = keras.Sequential([
    #(-1,32,32,3)->(-1,32,32,16)
    keras.layers.Conv2D(input_shape=(32, 32, 3),filters=32,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,32,32,32)->(-1,32,32,32)
    keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,32,32,32)->(-1,16,16,32)
    keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,16,16,32)->(-1,16,16,64)
    keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,16,16,64)->(-1,16,16,64)
    keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,16,16,64)->(-1,8,8,64)
    keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,8,8,64)->(-1,8*8*128)
    keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,8,8,128)->(-1,8*8*128)
    keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,8,8,128)->(-1,8*8*128)
    keras.layers.Flatten(),
    #(-1,8*8*128)->(-1,256)
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    #(-1,256)->(-1,10)
    keras.layers.Dense(10, activation="softmax")
])

print(model.summary())

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size = batch_size, epochs=epochs, validation_data=(test_images[:1000],test_labels[:1000]))

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(np.argmax(model.predict(test_images[:20]),1),test_labels[:20])


