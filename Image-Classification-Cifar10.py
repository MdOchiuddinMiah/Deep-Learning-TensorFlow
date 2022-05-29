import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import datetime


def plot_sample(index):
    plt.figure(figsize=(10, 1))
    plt.imshow(x_train[index])
    plt.show()


def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32, 3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


devices = tf.config.experimental.list_physical_devices()
tensor_version = tf.__version__
print(devices)
print(tensor_version)
print(tf.test.is_built_with_cuda())

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape)
print(y_train.shape)

plot_sample(0)

# Prepossing image data
x_train_scaled = x_train / 255
x_test_scaled = x_test / 255

y_train_categorical = keras.utils.to_categorical(
    y_train, num_classes=10, dtype='float32'
)
y_test_categorical = keras.utils.to_categorical(
    y_test, num_classes=10, dtype='float32'
)

startTime = datetime.datetime.now()
model = get_model()
model.fit(x_train_scaled, y_train_categorical, epochs=10)
endTime = datetime.datetime.now()
duration = endTime - startTime
print('Total minutes = {}'.format(duration.total_seconds()/60.0))
