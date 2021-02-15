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

    def __init__(self, input_dim=4, alpha=0.01):
        self.input_dim = input_dim
        self.w = np.random.uniform(size=(input_dim, ))
        self.b = 0
        self.alpha = alpha

    def train(self, x, y):
        pred = y * (np.dot(x, self.w) + self.b)
        if pred <= 0:
            self.w += self.alpha * y * x
            self.b += self.alpha * y

    def pred(self, x):
        return np.dot(x, self.w) + self.b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.01, help="learning rate")
    parser.add_argument('--epoch', type=int, default=10, help="epoch")
    args = parser.parse_args()

    logging.info('args: %s', args)

    x_train, y_train, x_test, y_test = load_data()
    for i, x in enumerate(x_train[:10]):
        print(i, x, y_train[i])

    model = Perceptron(input_dim=len(x_train[0]), alpha=args.alpha)
    for epoch in range(args.epoch):
        for i, x in enumerate(x_train):
            y = y_train[i]
            model.train(x, y)

        y_pred_v = model.pred(x_test)
        y_pred = np.sign(y_pred_v)
        acc = accuracy_score(y_test, y_pred)
        logging.info('epoch: %s, accuracy: %s', epoch + 1, acc)

    logging.info('report')
    p = classification_report(y_test, y_pred)
    print(p)


if __name__ == "__main__":
    main()
