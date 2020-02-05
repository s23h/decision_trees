import numpy as np
import scipy
import math
from scipy import stats
import sys
import pickle as pi
from collections import deque

class Dataset(object):

    def __init__(self, filename=None):
        self.features = None
        self.labels = None
        if filename:
            self.read(filename)

    def read(self, filename):
        with open(filename, 'r') as f:
            n = len(f.readline().split(','))
        self.features = np.genfromtxt(filename, delimiter=',', encoding=None,
                                      usecols=list(range(0, n-1)), dtype='float')
        self.labels = np.genfromtxt(filename, delimiter=',', encoding=None,
                                    usecols=[n-1], dtype=None)

    def find_maxmin(self):
        ymaxs = np.max(self.features, axis=0)
        ymins = np.min(self.features, axis=0)
        return ymaxs, ymins

    def label_distribution(self):
        return np.unique(self.labels, return_counts=True)


class DecisionNode(object):

    def __init__(self, split_col=None, split_val=None, prediction=None, label=None):
        self.split_col = split_col
        self.split_val = split_val
        self.left = None
        self.right = None
        self.prediction = prediction
        self.label = label

    def __str__(self):
        return "Decision: {0} < {1}".format(self.label, self.split_val)


class LeafNode(DecisionNode):

    def __str__(self):
        return "Prediction: {0}".format(self.prediction)


class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X

    """

    def __init__(self, headers=None):
        self.is_trained = False
        self.root = None
        self.headers = {i: h for i, h in enumerate(headers)} if headers else None

    def calc_probabilities(self, y):
        vals, cnts = np.unique(y, return_counts=True)
        probs = cnts.astype(float) / len(y)
        return probs

    def find_opt_split(self, x, y):
        min_ent = float('inf')
        opt_val, opt_col_idx = None, None
        opt_left, opt_right = None, None

        classes = np.unique(y)
        if len(classes) == 1:
            return None

        for i, col in enumerate(x.T):
            for val in col:
                left_y, right_y = self.split(i, val, x, y)
                ent_y = stats.entropy(self.calc_probabilities(left_y))
                ent_r = stats.entropy(self.calc_probabilities(right_y))
                split_ent = (len(left_y) * ent_y + len(right_y) * ent_r) / len(y)

                if split_ent < min_ent:
                    min_ent = split_ent
                    opt_val, opt_col_idx = val, i
                    opt_left, opt_right = left_y, right_y

        return opt_left, opt_right, opt_val, opt_col_idx

    def split(self, idx, val, x, data):
        left, right = [], []
        for i, item in enumerate(x.T[idx]):
            if item < val:
                left.append(data[i])
            else:
                right.append(data[i])
        return np.asarray(left), np.asarray(right)

    def train(self, x, y):
        """ Constructs a decision tree classifier from data

        Parameters
        ----------
        x : numpy.array
        An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        def train_rec(x, y):
            found_split = self.find_opt_split(x, y)
            if not found_split:
                return LeafNode(prediction=y[0])

            left_y, right_y, opt_val, opt_col_idx = found_split
            left_x, right_x = self.split(opt_col_idx, opt_val, x, x)

            if self.headers:
                node = DecisionNode(split_col=opt_col_idx, split_val=opt_val,
                                    label=self.headers[opt_col_idx])
            else:
                node = DecisionNode(split_col=opt_col_idx, split_val=opt_val,
                                    label="Col {0}".format(op_col_idx))

            node.left = train_rec(left_x, left_y)
            node.right = train_rec(right_x, right_y)

            return node

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        self.root = train_rec(x, y)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        def predict_sample(sample):
            node = self.root
            while not node.prediction:
                if sample[node.split_col] < node.split_val and node.left:
                    node = node.left
                elif node.right:
                    node = node.right
            return node.prediction

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        predictions = [predict_sample(sample) for sample in x]
        return np.asarray(predictions)

    def serialise_model(self, filename):
        with open(filename, 'wb') as f:
            pi.dump(self.root, f)

    def deserialise_model(self, filename):
        with open(filename, 'rb') as f:
            self.root = pi.load(f)
        self.is_trained = True

    def __str__(self):
        if not self.is_trained:
            return "Untrained Model"

        space, branch, tee, last =  '    ', '|   ', '+-- ', '\'-- '

        def display_rec(node, prefix=''):
            if node is self.root:
                yield node.__str__()
            contents = (node.left, node.right)
            pointers = (tee, last)
            for pointer, node in zip(pointers, contents):
                yield prefix + pointer + node.__str__()
                if not node.prediction: # extend the prefix and recurse:
                    extension = branch if pointer == tee else space
                    yield from display_rec(node, prefix=prefix+extension)

        return "\n".join((line for line in display_rec(self.root)))


if __name__ == "__main__":

    from classification import *

    headers = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
               "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

    if len(sys.argv) == 3 and sys.argv[1] == "train":
        ds = Dataset(sys.argv[2])
        model = DecisionTreeClassifier(headers)
        model.train(ds.features, ds.labels)
        model.serialise_model("data/model_sub.pickle")
    else:
        model = DecisionTreeClassifier(headers)
        model.deserialise_model("data/model_sub.pickle")
        print(model)
        valid = Dataset()
        valid.read("data/validation.txt")
        preds = model.predict(valid.features)
        count = 0
        for i, pred in enumerate(preds):
            # print(pred, valid.labels[i])
            if pred == valid.labels[i]:
                count += 1
        print((float(count) / float(len(valid.labels)) * 100))
