import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import random

random.randint(0, 6)


def batch_gradient_descent(X, y_true, epochs, learning_rate=0.01):
    number_of_features = X.shape[1]
    w = np.ones(shape=number_of_features)
    b = 0
    total_samples = X.shape[0]

    cost_list = []
    epoch_list = []

    for i in range(epochs):
        y_predicted = np.dot(w, X.T) + b

        w_grad = -(2 / total_samples) * (X.T.dot(y_true - y_predicted))
        b_grad = -(2 / total_samples) * np.sum(y_true - y_predicted)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        cost = np.mean(np.square(y_true - y_predicted))

        if i % 10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    return w, b, cost, cost_list, epoch_list


def stochastic_gradient_descent(x, y_true, epochs, learning_rate=0.01):
    number_of_features = x.shape[1]
    w = np.ones(shape=number_of_features)
    b = 0
    total_samples = x.shape[0]

    cost_list = []
    epoch_list = []

    for i in range(epochs):
        random_index = random.randint(0, total_samples - 1)
        sample_x = x[random_index]
        sample_y = y_true[random_index]

        y_predicted = np.dot(w, sample_x.T) + b

        w_grad = -(2 / total_samples) * (sample_x.T.dot(sample_y - y_predicted))
        b_grad = -(2 / total_samples) * (sample_y - y_predicted)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        cost = np.square(sample_y - y_predicted)

        if i % 100 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    return w, b, cost, cost_list, epoch_list


def predict(area, bedrooms, w, b):
    scaled_X = sx.transform([[area, bedrooms]])[0]
    scaled_price = w[0] * scaled_X[0] + w[1] * scaled_X[1] + b
    return sy.inverse_transform([[scaled_price]])[0][0]


def plotResult(name, epoch_list, cost_list):
    plt.xlabel("epoch " + name)
    plt.ylabel("cost " + name)
    plt.plot(epoch_list, cost_list)
    plt.show()


def printValues(name, w, b, cost):
    # print('W1=', w[0], 'W2=', w[1], 'Bias=', b, 'Cost=', cost)
    print(name + ' W1= {} W2= {} Bias= {} Cost= {}'.format(w[0], w[1], b, cost))


df = pd.read_csv("C:\\Sayed\\Python\\py-master-codebasics\\DeepLearningML\\8_sgd_vs_gd\\homeprices_banglore.csv")
sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()

# print(df.sample(5))
print(df.shape)
# print(df.shape[0])
# print(df.shape[1])
scaled_x = sx.fit_transform(df.drop('price', axis='columns'))
scaled_y = sy.fit_transform(df['price'].values.reshape(df.shape[0], 1))
# print(scaled_x[:5])
# print(scaled_y[:5])
#
# test1 =scaled_y.reshape(scaled_y.shape[0],)
# test2 = scaled_x.T
# print('1D array')
# print(test1[:5])
# print(test2[:5])

w, b, cost, cost_list, epoch_list = batch_gradient_descent(scaled_x, scaled_y.reshape(scaled_y.shape[0], ), 500)
plotResult('BGD', epoch_list, cost_list)
printValues('BGD', w, b, cost)
print('BGD Predicted value {p}'.format(p=predict(2600, 4, w, b)))

w_sgd, b_sgd, cost_sgd, cost_list_sgd, epoch_list_sgd = stochastic_gradient_descent(scaled_x, scaled_y.reshape(
    scaled_y.shape[0], ), 10000)
plotResult('SGD', epoch_list_sgd, cost_list_sgd)
printValues('SGD', w_sgd, b_sgd, cost_sgd)
print('SGD Predicted value {p}'.format(p=predict(2600, 4, w_sgd, b_sgd)))
