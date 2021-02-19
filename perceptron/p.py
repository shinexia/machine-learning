import logging
import argparse
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s')


def load_data():
    iris = load_iris()
    xs = iris.data
    ys = iris.target
    idx = ys != 1
    xs = xs[idx]
    ys = ys[idx] - 1
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)
    logging.info('x_train: %s, y_train: %s, x_test: %s, y_test: %s', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


class Perceptron(object):

    def __init__(self, x_train, y_train, input_dim=4, alpha=0.01):
        self.input_dim = input_dim
        self.w = np.zeros((input_dim, ))
        self.b = 0
        self.alpha = alpha
        self.x_train = x_train
        self.y_train = y_train

    def train(self, xi):
        x, y = self.x_train[xi], self.y_train[xi]
        pred = y * (np.dot(x, self.w) + self.b)
        if pred <= 0:
            self.w += self.alpha * y * x
            self.b += self.alpha * y

    def pred(self, x):
        logging.info('w: %s', self.w)
        return np.dot(x, self.w) + self.b


class PerceptronDual(object):

    def __init__(self, x_train, y_train, input_dim=4, alpha=0.01):
        self.input_dim = input_dim
        self.num_train = len(x_train)
        self.a = np.zeros((self.num_train, ))
        self.b = 0
        self.alpha = alpha
        self.gram = np.zeros((self.num_train, self.num_train))
        self.x_train = x_train
        self.y_train = y_train
        for i, xi in enumerate(x_train):
            for j, xj in enumerate(x_train):
                self.gram[i][j] = np.dot(xi, xj) * y_train[i]

    def train(self, xi):
        y = self.y_train[xi]
        s = self.b
        for j in range(self.num_train):
            s += self.gram[j][xi] * self.a[j]
        pred = y * s
        if pred <= 0:
            self.a[xi] += self.alpha
            self.b += self.alpha * y

    def pred(self, x):
        w = np.sum(self.a.reshape(-1, 1) * self.x_train * self.y_train.reshape(-1, 1), axis=0)
        logging.info('w: %s', w)
        return np.dot(x, w) + self.b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='', help="")
    parser.add_argument('--alpha', type=float, default=0.01, help="learning rate")
    parser.add_argument('--epoch', type=int, default=10, help="epoch")
    args = parser.parse_args()

    logging.info('args: %s', args)

    x_train, y_train, x_test, y_test = load_data()
    for i, x in enumerate(x_train[:10]):
        print(i, x, y_train[i])

    clazz = PerceptronDual if args.mode == 'dual' else Perceptron
    model = clazz(x_train, y_train, input_dim=len(x_train[0]), alpha=args.alpha)
    for epoch in range(args.epoch):
        for i in range(len(x_train)):
            model.train(i)

        y_pred_v = model.pred(x_test)
        y_pred = np.sign(y_pred_v)
        acc = accuracy_score(y_test, y_pred)
        logging.info('epoch: %03d, accuracy: %s', epoch + 1, acc)

    logging.info('report')
    p = classification_report(y_test, y_pred)
    print(p)


if __name__ == "__main__":
    main()
