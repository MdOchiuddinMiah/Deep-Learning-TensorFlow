import numpy as np
# import tensorflow as tf
# from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class DataRetrive:
    filepath = "C:\\Sayed\\ML Data\\insurance_data.csv"  # class variable
    feature_col = ['age', 'affordibility']
    class_col = "bought_insurance"
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    def __init__(self, localPath, feature_col, class_col):
        self.localPath = localPath
        self.feature_col = feature_col
        self.class_col = class_col

    @classmethod
    def getPath(cls):
        return cls.filepath

    @classmethod
    def getFeatuteCol(cls):
        return cls.feature_col

    @classmethod
    def getClassCol(cls):
        return cls.class_col

    def getData(self):
        return pd.read_csv(self.filepath)

    def dataSplit(self, data):
        x_train, x_test, y_train, y_test = train_test_split(data[self.feature_col], data[self.class_col],
                                                            train_size=0.2, random_state=25)
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        return x_train, x_test, y_train, y_test

    def getDataType(self, data, col=None):
        return col is None and data.dtypes or data[col].dtype


class NeuralHelper:
    def __init__(self):
        pass

    @classmethod
    def getSigmoid(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def getLogLoss(cls, y_true, y_predicted):
        epsilon = 1e-15
        y_predicted_new = [max(i, epsilon) for i in y_predicted]
        y_predicted_new = [min(i, 1 - epsilon) for i in y_predicted_new]
        y_predicted_new = np.array(y_predicted_new)
        return -np.mean(y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new))


class NeuralClassifier:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0

    #     utility methods
    def gradient_descent(self, age, affordability, y_true, epochs, loss_thresold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordability + bias
            y_predicted = NeuralHelper.getSigmoid(weighted_sum)
            loss = NeuralHelper.getLogLoss(y_true, y_predicted)

            w1d = (1 / n) * np.dot(np.transpose(age), (y_predicted - y_true))
            w2d = (1 / n) * np.dot(np.transpose(affordability), (y_predicted - y_true))

            bias_d = np.mean(y_predicted - y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d

            if i % 50 == 0:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

            if loss <= loss_thresold:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias

    def fit(self, x, y, epochs, loss_thresold):
        self.w1, self.w2, self.bias = self.gradient_descent(x['age'], x['affordibility'], y, epochs, loss_thresold)
        print(f"Final weights and bias: w1: {self.w1}, w2: {self.w2}, bias: {self.bias}")

    def predict(self, x_test):
        weighted_sum = self.w1 * x_test['age'] + self.w2 * x_test['affordibility'] + self.bias
        sigmoid_result = NeuralHelper.getSigmoid(weighted_sum)
        result = []
        for item in sigmoid_result:
            if item >= 0.5:
                result.append(1)
            else:
                result.append(0)

        return result


# call class
dataRetrive = DataRetrive(DataRetrive.getPath(), DataRetrive.getFeatuteCol(), DataRetrive.getClassCol())
filedata = dataRetrive.getData()
print(filedata.head())
x_train, x_test, y_train, y_test = dataRetrive.dataSplit(filedata)

x_train_scaled = x_train.copy()
x_train_scaled['age'] = x_train_scaled['age'] / 100

x_test_scaled = x_test.copy()
x_test_scaled['age'] = x_test_scaled['age'] / 100

# call neural network
neuralNetModel = NeuralClassifier()
neuralNetModel.fit(x_train_scaled, y_train, epochs=8000, loss_thresold=0.01)
predicted = neuralNetModel.predict(x_test_scaled)
original_predicted = y_test.tolist()
model_predicted = predicted
confusion = confusion_matrix(original_predicted, model_predicted)
print(confusion)
print(classification_report(original_predicted, model_predicted))
