import numpy as np
import scipy
import math
from scipy import stats
import sys
import pickle as pi

class Dataset(object):

    def __init__(self):
        self.features = None
        self.labels = None

    def read(self, filename):
        with open(filename, 'r') as f:
            n = len(f.readline().split(','))
        self.features = np.genfromtxt(filename, delimiter=',', usecols=list(range(0, n-1)), dtype='float')
        self.labels = np.genfromtxt(filename, delimiter=',', usecols=[n-1], dtype=None)

    def find_maxmin(self):
        ymaxs = np.max(self.features, axis=0)
        ymins = np.min(self.features, axis=0)
        return ymaxs, ymins

    def label_distribution(self):
        return np.unique(self.labels, return_counts=True)


class DecisionNode(object):

    def __init__(self, split_col=None, split_val=None, prediction=None):
        self.split_col = split_col
        self.split_val = split_val
        self.left = None
        self.right = None
        self.prediction = prediction


class LeafNode(DecisionNode):

    def f():
        return


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

    def __init__(self, filename):
        self.is_trained = False
        self.filename = filename
        self.root = None

    # def calc_entropy(self, y):
    #     num_lbls = len(y)
    #
    #     if num_lbls <= 1:
    #         return 0
    #
    #     val, cnts = np.unique(y, return_counts = True)
    #     probs = cnts.astype(float) / num_lbls
    #     num_cl = np.count_nonzero(probs)
    #
    #     if  num_cl <= 1:
    #         return 0
    #
    #     entropy = 0
    #
    #     for prob in probs:
    #         entropy -= prob * math.log(prob, 2)
    #
    #     return entropy

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

    def split(self, idx, val, x, y=None):
        left, right = [], []
        y_flag = (y is not None)
        for i, item in enumerate(x.T[idx]):
            if item < val:
                left.append(y[i] if y_flag else x[i])
            else:
                right.append(y[i] if y_flag else x[i])
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
            opt_split = self.find_opt_split(x, y)
            if not opt_split:
                return LeafNode(prediction=y[0])

            left_y, right_y, opt_val, opt_col_idx = opt_split
            left_x, right_x = self.split(opt_col_idx, opt_val, x)

            node = DecisionNode(opt_col_idx, opt_val)

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

        def predict_rec(x, node):
            if not node.prediction:
                if x[node.split_col] < node.split_val and node.left:
                    pred = predict_rec(x, node.left)
                elif node.right:
                    pred = predict_rec(x, node.right)
                return pred
            else:
                return node.prediction

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        predictions = []
        for sample in x:
            predictions = np.append(predictions, predict_rec(sample, self.root))

        return predictions

    def serialise_model(self):
        with open(self.filename, 'wb') as f:
            pi.dump(self.root, f)

    def deserialise_model(self):
        with open(self.filename, 'rb') as f:
            self.root = pi.load(f)
        self.is_trained = True

    def traverse(self, curr):
        if not curr.prediction:
            desc = "Node("
            desc += self.traverse(curr.left) if curr.left else "empty"
            desc  += ", "
            desc += self.traverse(curr.right) if curr.right else "empty"
            return desc + ")"
        else:
            return str(curr.prediction)

dst = Dataset()

def setup(ds, filename):
    ds.read(filename)

    model = DecisionTreeClassifier("data/model.pickle")
    model.train(ds.features, ds.labels)
    model.serialise_model()

    return model

if len(sys.argv) == 3 and sys.argv[1] == "train":
    model = setup(dst, sys.argv[2])
else:
    model = DecisionTreeClassifier("data/model.pickle")
    model.deserialise_model()
    valid = Dataset()
    valid.read("data/validation.txt")
    preds = model.predict(valid.features)
    count = 0
    for i, pred in enumerate(preds):
        print(pred, valid.labels[i])
        if pred == valid.labels[i]:
            count += 1
    print((float(count) / float(len(valid.labels)) * 100))
