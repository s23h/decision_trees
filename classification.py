##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the
# DecisionTreeClassifier
##############################################################################

import numpy as np
import scipy
from scipy import stats

class Dataset(object):

    def __init__(self):
        self.features = None
        self.labels = None

    def read(self, filename):
        with open(filename, 'r') as f:
            n = len(f.readline().split(','))
        self.features = np.loadtxt(filename, delimiter=',', usecols=(range(0, n-1)), dtype='float')
        self.labels = np.loadtxt(filename, delimiter=',', usecols=(n-1), dtype='S12')

    def find_maxmin(self):
        ymaxs = np.max(self.features, axis=0)
        ymins = np.min(self.features, axis=0)
        return ymaxs, ymins

    def label_distribution(self):
        return np.unique(self.labels, return_counts=True)

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

    split_col : int
        Keeps track of the column along which we partioned the dataset

    split_val : int
        Keeps track of the value which we used to split the column - the left split
        will contain everything smaller than split_bound and the right split will
        contain everything larger than the split bound.

    left : DecisionTreeClassifier
        The left partion of the dataset will be evaluated along this path.

    right : DecisionTreeClassifier
        The right partition of the dataset will be evaluated along this path.
    """

    def __init__(self, split_col=None, split_val=None):
        self.is_trained = False
        self.split_col = split_col
        self.split_val = split_val
        self.left = None
        self.right = None
        self.prediction = -1

    def calc_entropy(self, y):
        num_lbls = len(y)

        if num_lbls <= 1:
            return 0

        val, cnts = np.unique(y, return_counts = True)
        probs = cnts / num_lbls
        num_cl = np.count_nonzero(probs)

        if  num_cl <= 1:
            return 0

        entropy = 0

        for prob in probs:
            entropy -= prob * log(prob, 2)

        return entropy

    def optimal_col_split(self, col, y):
        opt_left_y, opt_right_y = np.array([]), np.array([])
        min_ent = float('inf')
        split_val = -1

        for candidate in np.unique(col):

            left_y, right_y = np.array([]), np.array([])

            for i in range(len(col)):
                if col[i] < candidate:
                    np.append(left_y, y[i])
                else:
                    np.append(right_y, y[i])

            split_ent = len(left_y)/len(y) * self.calc_entropy(left_y) + len(right_y)/len(y) * self.calc_entropy(right_y)

            if split_ent < min_ent:
                min_ent = split_ent
                split_val = candidate
                opt_left_y = left_y
                opt_right_y = right_y

        return min_ent, split_val, opt_left_y, opt_right_y

    def optimal_split(self, x, y):
        min_ent = float('inf')
        opt_split_val = -1
        opt_split_col = -1
        y_left, y_right = np.array([]), np.array([])

        for i in range(len(x.T)):
            split_ent, split_val, l_y, r_y = self.optimal_col_split(x.T[i], y)

            if split_ent < min_ent:
                min_ent = split_ent
                opt_split_val = split_val
                opt_split_col = i
                y_left = l_y
                y_right = r_y

        x_left, x_right = np.array([]), np.array([])

        for i in range(len(x.T)):
            if x.T[opt_split_col][i] < opt_split_val:
                np.append(x_left, x[i])
            else:
                np.append(x_right, x[i])

        return opt_split_col, opt_split_val, x_left, y_left, x_right, y_right

    def train(self, x, y):
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        split_col, split_val, x_l, y_l, x_r, y_r = self.optimal_split(x, y)
        self.split_col = split_col
        self.split_val = split_val

        self.left = DecisionTreeClassifier()

        if self.calc_entropy(y_l) == 0 and len(y_l) > 0:
            self.left.prediction = y_l[0]
        else:
            self.left.train(x_l, y_l)

        self.right = DecisionTreeClassifier()

        if self.calc_entropy(y_r) == 0 and len(y_r) > 0:
            self.right.prediction = y_r[0]
        else:
            self.right.train(x_r, y_r)

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

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)


        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################


        # remember to change this if you rename the variable
        return predictions


        #######################################################################
        #                 **            UTILITIES           **
        #######################################################################



def gof(d1_vals, d1_counts, d2_vals, d2_counts):
    o = scipy.array(d2_counts)
    o_sum = sum(d2_counts)
    e = scipy.array(d1_counts)
    e_sum = sum(d1_counts)

    for i, exp in enumerate(e):
        e[i] = (exp / e_sum) * o_sum

    print(stats.chisquare(o, f_exp=e))

def differ(d1_vals, d1_counts, d2_vals, d2_counts):
    d2_map = {val.decode("utf-8"): count for val, count in zip(d2_vals, d2_counts)}
    res = []

    for val, count in zip(d1_vals, d1_counts):
        val = val.decode("utf-8")
        res.append((val, count, d2_map[val], (count - d2_map[val])*100 / count))

    print(res)


        #######################################################################
        #                 **             MAIN             **
        #######################################################################

'''
d1 = Dataset()
d1.read("data/train_full.txt")
d1_vals, d1_counts = d1.label_distribution()

d2 = Dataset()
d2.read("data/train_noisy.txt")
d2_vals, d2_counts = d2.label_distribution()

gof(d1_vals, d1_counts, d2_vals, d2_counts)
differ(d1_vals, d1_counts, d2_vals, d2_counts)
'''

ds = Dataset()
ds.read("data/toy.txt")

model = DecisionTreeClassifier()
model.train(ds.features, ds.labels)
