import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz


def accuracy(policy, obss, acts):
    return np.mean(acts == policy.pick_action(obss))


def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test


def save_dt_policy(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    f = open(dirname + '/' + fname, 'wb')
    pickle.dump(dt_policy, f)
    f.close()


def save_dt_policy_viz(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    file = dirname + '/' + fname
    export_graphviz(dt_policy.tree, file)
    with open(file) as dot_file:
        dot_data = dot_file.read()
    graph = graphviz.Source(dot_data)
    graph.render(file)


def load_dt_policy(dirname, fname):
    f = open(dirname + '/' + fname, 'rb')
    dt_policy = pickle.load(f)
    f.close()
    return dt_policy


class DTPolicy:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        self.fit(obss_train, acts_train)
        # print('Train accuracy: {}'.format(accuracy(self, obss_train, acts_train)))
        # print('Test accuracy: {}'.format(accuracy(self, obss_test, acts_test)))
        print('Number of nodes: {}'.format(self.tree.tree_.node_count))

    def pick_action(self, obss, on_training=0):
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth)
        clone.tree = self.tree
        return clone


if __name__ == '__main__':
    tree = DecisionTreeClassifier(max_depth=10)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 10, 100)
    tree.fit(X, y)
    print(tree.predict(X).shape)


