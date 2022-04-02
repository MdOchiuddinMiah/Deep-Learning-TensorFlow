import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import cv2

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# my written test
file = r'C:\\Users\\BS-526\\Pictures\\Camera Roll\\9_test.png'
test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
# plt.imshow(test_image, cmap='gray')
# plt.show()

# format image
resized_image = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
resized_image = cv2.bitwise_not(resized_image)
plt.matshow(resized_image)
plt.show()

# print(resized_image)

# scale the value from 0-1 for enhance performance; max pixel is 255
x_train = x_train / 255
x_test = x_test / 255
resized_image= resized_image / 255

print(len(x_train), len(x_test))
print(x_train[0].shape)
print(x_train.shape)
print(y_train.shape)
print(resized_image.shape)
print(resized_image)

# show array value
# print(x_train[0])
# plt.matshow(x_train[0])
# plt.show()

# 28*28 array to one dimentional array
x_train_flattened = x_train.reshape(len(x_train), 28 * 28)
x_test_flattened = x_test.reshape(len(x_test), 28 * 28)
resized_image_flattened= resized_image.reshape(-1)
print(x_train_flattened.shape)
print(x_test_flattened.shape)
print(resized_image_flattened.shape)

# without hidden layer
# model = keras.Sequential([
#     keras.layers.Dense(10, input_shape=(28 * 28,), activation='sigmoid')
# ])

# without reshape we pass it in keras
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dense(10, activation='sigmoid')
# ])

# with hidden layer
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)
model.evaluate(x_test_flattened, y_test)

y_predicted = model.predict(x_test_flattened)
resized_image_predicted = model.predict(np.expand_dims(resized_image_flattened, axis=0))

# plt.matshow(x_test[0])
# plt.show()
print('My Image result= ')
print(np.argmax(resized_image_predicted))
print('Array First Image result= ')
print(np.argmax(y_predicted[0]))
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels[:5])

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
print(cm)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
