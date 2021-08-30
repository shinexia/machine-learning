import logging
import argparse
import numpy as np

from collections import Counter

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s')


def load_data():
    iris = load_iris()
    xs = iris.data
    ys = iris.target
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)
    logging.info('x_train: %s, y_train: %s, x_test: %s, y_test: %s',
                 x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


class TreeNode(object):

    def __init__(self, col, split_value, left=None, right=None, label=None):
        self.col = col
        self.split_value = split_value
        self.left = left
        self.right = right
        self.label = label

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class CARTModel(object):

    def __init__(self, tree: TreeNode = None):
        self.tree = tree


def gini(y_train):
    c = Counter(y_train)
    c = np.array(list(c.values()))
    c = c / np.sum(c)
    return 1 - np.sum(c * c)


class CART(object):

    def __init__(self):
        pass

    def train(self, x_train, y_train):
        pass

    def _select_best_split(self, x_train, y_train):
        pass

    def _split(self, x_train, y_train):
        pass

    def _build_tree(self, x_train, y_train):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
