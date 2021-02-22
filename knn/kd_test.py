import logging
import numpy as np

from . import kd


def test_split():
    xs = np.random.uniform(size=(10, ))
    p, left, right = kd.split(xs)
    logging.info('\nxs: %s\np: %s\nleft: %s\nright: %s', xs, xs[p], xs[left], xs[right])


def test_build():
    x_train = np.random.uniform(size=(20, 4))
    y_train = np.rint(np.random.uniform(size=(20,)))
    kdtree = kd.create(x_train, y_train)
    logging.info('kdtree: \n%s', kd.stringify_kdnode(kdtree.root))

    ret = kdtree.search(x_train[1], k=5)
    for dist, node in ret:
        logging.info('dist: %s, node: %s', dist, node)


def test_pred():
    x_train, y_train, x_test, y_test = kd.load_data()
    kdtree = kd.create(x_train=x_train, y_train=y_train)

    y_pred = kdtree.pred(x_test, k=3)

    logging.info('\ny_test: %s\ny_pred: %s', y_test, y_pred)

    p = kd.classification_report(y_test, y_pred)
    logging.info('report: \n%s', p)
