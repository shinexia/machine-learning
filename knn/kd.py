from __future__ import annotations

import logging
import argparse
import heapq
import numpy as np

from functools import cmp_to_key
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s')


def load_data():
    iris = load_iris()
    xs = iris.data
    ys = iris.target.astype(np.int)
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)
    logging.info('x_train: %s, y_train: %s, x_test: %s, y_test: %s', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


class Node():

    def __init__(self, data=None, left: Node = None, right: Node = None):
        self.data = data
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class KDNode(Node):

    def __init__(self, data=None, left: Node = None, right: Node = None, axis=None, label=None):
        super(KDNode, self).__init__(data=data, left=left, right=right)
        self.value = data[axis]
        self.axis = axis
        self.label = label

    def __repr__(self):
        return '{"data":' + '[' + ','.join(str(r) for r in self.data) + ']' + \
            ',"value": ' + str(self.value) + \
            ',"label":' + (str(self.label) if self.label is not None else 'null') + \
            ',"axis": ' + (str(self.axis) if self.axis is not None else 'null') + \
            '}'

    def __str__(self):
        return self.__repr__()


def stringify_kdnode(node: KDNode) -> str:
    if node is None:
        return 'null'
    return '{"data":' + '[' + ','.join(str(r) for r in node.data) + ']' + \
        ',"left": ' + stringify_kdnode(node.left) + \
        ',"right": ' + stringify_kdnode(node.right) + \
        ',"value": ' + str(node.value) + \
        ',"label":' + (str(node.label) if node.label is not None else 'null') + \
        ',"axis": ' + (str(node.axis) if node.axis is not None else 'null') + \
        '}'


class Heap():

    def __init__(self, k):
        self.k = k
        self.nodes = []
        self.max_dist = 0
        self.size = 0

    def add(self, node, dist):
        if self.size >= self.k:
            heapq.heapreplace(self.nodes, (-dist, id(node), node))
        else:
            heapq.heappush(self.nodes, (-dist, id(node), node))
            self.size += 1
        self.max_dist = -self.nodes[0][0]
        return True

    def result(self):
        ret = []
        for i in range(self.size):
            dist, _, node = heapq.heappop(self.nodes)
            ret.append((-dist, node))
        ret.reverse()
        return ret

    @property
    def is_full(self):
        return self.size >= self.k


class KDTree():

    def __init__(self, root: KDNode, dist_func=None):
        self.root: KDNode = root
        self.dist_func = dist_func

    def search(self, x, k=None) -> List[(float, KDNode)]:
        heap = Heap(k=k)
        self._search(heap, self.root, x)
        return heap.result()

    def _search(self, heap: Heap, node: KDNode, x):
        dist = self.dist_func(node.data, x)
        if not heap.is_full or dist < heap.max_dist:
            heap.add(node, dist)
        aixs_value = x[node.axis]
        best, other = None, None
        if aixs_value <= node.value:
            if node.left is not None:
                best = node.left
            if node.right is not None:
                if best is None:
                    best = node.right
                else:
                    other = node.right
        else:
            if node.right is not None:
                best = node.right
            if node.left is not None:
                if best is None:
                    best = node.left
                else:
                    other = node.left
        if best is not None:
            self._search(heap, best, x)
        if other is not None and (not heap.is_full or self.dist_func(node.value, aixs_value) < heap.max_dist):
            self._search(heap, other, x)

    def pred(self, xs, k=1):
        return np.array([self._pred(x, k=k) for x in xs], np.int)

    def _pred(self, x, k=1):
        rs = self.search(x, k=k)
        if k == 1:
            return rs[0][1].label
        stat = {}
        for dist, node in rs:
            label = node.label
            s = stat.get(label, None)
            if s is None:
                s = dict(label=label, count=1, dist=dist)
                stat[label] = s
            else:
                s['count'] += 1
                s['dist'] += dist
        labels = list(stat.values())

        def compare(a, b):
            if a['count'] > b['count']:
                return -1
            elif a['count'] < b['count']:
                return 1
            elif a['dist'] > b['dist']:
                return 1
            elif a['dist'] < b['dist']:
                return -1
            else:
                return 0
        labels.sort(key=cmp_to_key(compare))
        r = labels[0]['label']
        logging.info('r: %s, labels: %s', r, labels)
        return r


def l2_distance(a, b):
    return np.linalg.norm(a - b)


def split(xs):
    """
    Args:
        xs: 1-d array
    Return:
        (p, left, right), index of xs
    """
    num_left_min = int((len(xs) - 1) / 2)
    num_left_max = num_left_min if len(xs) & 0x1 == 1 else (num_left_min + 1)
    left, right, idx = [], [], list(range(len(xs)))
    p = 0
    while True:
        ls, rs = [], []
        pv = xs[p]
        for i in idx:
            if i == p:
                continue
            if xs[i] <= pv:
                ls.append(i)
            else:
                rs.append(i)
        num_left = len(left) + len(ls)
        if num_left < num_left_min:
            left.extend(ls)
            left.append(p)
            p = rs[0]
            idx = rs
        elif num_left > num_left_max:
            right.append(p)
            right.extend(rs)
            p = ls[0]
            idx = ls
        else:
            left.extend(ls)
            right.extend(rs)
            return p, left, right


def _create_kdnode(x_train, y_train, axis=0, dim=None) -> KDNode:
    if len(x_train) == 0:
        return None
    p, left, right = split(x_train[:, axis])
    left_node = _create_kdnode(x_train[left], y_train[left], (axis + 1) % dim, dim=dim)
    right_node = _create_kdnode(x_train[right], y_train[right], (axis + 1) % dim, dim=dim)
    return KDNode(data=x_train[p], left=left_node, right=right_node, axis=axis, label=y_train[p])


def create(x_train, y_train, dist_func=l2_distance) -> KDTree:
    dim = len(x_train[0])
    root = _create_kdnode(x_train, y_train, axis=0, dim=dim)
    return KDTree(root, dist_func=dist_func)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=1, help='')
    args = parser.parse_args()
    logging.info('args: %s', args)

    x_train, y_train, x_test, y_test = load_data()
    kdtree = create(x_train=x_train, y_train=y_train, dist_func=l2_distance)

    y_pred = kdtree.pred(x_test, k=args.k)
    logging.info('\ny_test: %s\ny_pred: %s', y_test, y_pred)

    p = classification_report(y_test, y_pred)
    logging.info('report: \n%s', p)


if __name__ == '__main__':
    main()
